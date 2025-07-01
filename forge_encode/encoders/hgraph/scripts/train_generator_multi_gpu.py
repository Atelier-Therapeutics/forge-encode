import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

import rdkit
import math, random, sys
import numpy as np
import argparse
import os
from tqdm.auto import tqdm

# Add the project root to the Python path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
sys.path.insert(0, project_root)

# Now import the modules
from forge_encode.encoders.hgraph.hgnn import HierVAE
from forge_encode.encoders.hgraph.vocab import PairVocab, common_atom_vocab
from forge_encode.encoders.hgraph.dataset import MoleculeDataset, DataFolder


lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--load_model', default=None)
parser.add_argument('--seed', type=int, default=7)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--depthT', type=int, default=15)
parser.add_argument('--depthG', type=int, default=15)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=5.0)
parser.add_argument('--step_beta', type=float, default=0.001)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)

parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=25000)
parser.add_argument('--print_iter', type=int, default=50)
parser.add_argument('--save_iter', type=int, default=5000)

# DDP specific arguments
parser.add_argument('--use_multi_gpu', action='store_true', help='Use all available GPUs with DDP')
parser.add_argument('--gpu_ids', type=str, default=None, help='Comma-separated list of GPU IDs to use (e.g., "0,1")')
parser.add_argument('--world_size', type=int, default=None, help='Number of GPUs to use (auto-detected if not specified)')
parser.add_argument('--rank', type=int, default=0, help='Rank of this process (for DDP)')
parser.add_argument('--dist_url', type=str, default='tcp://localhost:12355', help='URL for DDP initialization')

args = parser.parse_args()

def setup_ddp(rank, world_size, args):
    """Setup DDP environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set the device for this process
    torch.cuda.set_device(rank)
    
    print(f"DDP Process {rank}/{world_size} initialized on GPU {rank}")

def cleanup_ddp():
    """Cleanup DDP environment"""
    dist.destroy_process_group()

def train_worker(rank, world_size, args):
    """Worker function for each GPU process"""
    # Setup DDP
    setup_ddp(rank, world_size, args)
    
    # Set random seeds (same for all processes for reproducibility)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Load vocabulary
    vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
    args.vocab = PairVocab(vocab)
    
    # Create model and move to GPU
    model = HierVAE(args).cuda(rank)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    if rank == 0:  # Only print on main process
        print(f"DDP Training with {world_size} GPUs")
        print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(rank)}")
    
    # Initialize model parameters
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
    
    # Load checkpoint if specified
    if args.load_model:
        if rank == 0:
            print('continuing from checkpoint ' + args.load_model)
        checkpoint = torch.load(args.load_model, map_location=f'cuda:{rank}')
        model_state, optimizer_state, total_step, beta = checkpoint
        model.module.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    else:
        total_step = beta = 0
    
    # Helper functions
    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))
    
    # Training loop
    meters = np.zeros(6)
    for epoch in range(args.epoch):
        # Create dataset - each process will get different batches
        dataset = DataFolder(args.train, args.batch_size)
        
        # Create progress bar only on main process
        if rank == 0:
            pbar = tqdm(dataset, desc=f"Epoch {epoch+1}/{args.epoch}")
        else:
            pbar = dataset
        
        for batch in pbar:
            total_step += 1
            model.zero_grad()
            graphs, tensors, orders = batch
            
            # Add debug output to see batch size (only on main process)
            if total_step == 1 and rank == 0:
                print(f"DEBUG: First batch contains {len(orders)} molecules")
                print(f"DEBUG: Orders shape: {len(orders)}")
                if hasattr(orders[0], '__len__'):
                    print(f"DEBUG: First order length: {len(orders[0])}")
            
            loss, kl_div, wacc, iacc, tacc, sacc = model(graphs, tensors, orders, beta=beta)
            
            # Sum loss across GPUs if it's a tensor
            if loss.dim() > 0:
                loss = loss.sum()
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            
            # Update meters (only on main process for logging)
            if rank == 0:
                meters = meters + np.array([
                    kl_div.detach().cpu().sum().item() if torch.is_tensor(kl_div) else kl_div,
                    loss.item() if torch.is_tensor(loss) else loss,
                    wacc.detach().cpu().sum().item() if torch.is_tensor(wacc) else wacc,
                    iacc.detach().cpu().sum().item() if torch.is_tensor(iacc) else iacc,
                    tacc.detach().cpu().sum().item() if torch.is_tensor(tacc) else tacc,
                    sacc.detach().cpu().sum().item() if torch.is_tensor(sacc) else sacc
                ])
            
            # Print progress (only on main process)
            if rank == 0 and total_step % args.print_iter == 0:
                meters /= args.print_iter
                print("[%d] Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], param_norm(model), grad_norm(model)))
                sys.stdout.flush()
                meters *= 0
            
            # Save checkpoint (only on main process)
            if rank == 0 and total_step % args.save_iter == 0:
                model_state = model.module.state_dict()
                ckpt = (model_state, optimizer.state_dict(), total_step, beta)
                torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{total_step}"))
            
            # Update learning rate (only on main process)
            if rank == 0 and total_step % args.anneal_iter == 0:
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])
            
            # Update beta (only on main process)
            if rank == 0 and total_step >= args.warmup and total_step % args.kl_anneal_iter == 0:
                beta = min(args.max_beta, beta + args.step_beta)
        
        # Synchronize all processes at end of epoch
        dist.barrier()
    
    # Cleanup
    cleanup_ddp()

def main():
    """Main function to launch DDP training"""
    print(args)
    
    # Determine number of GPUs to use
    if args.use_multi_gpu:
        if args.gpu_ids:
            # Use specific GPU IDs
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
            world_size = len(gpu_ids)
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
            print(f"Using specific GPUs: {gpu_ids}")
        else:
            # Use all available GPUs
            world_size = torch.cuda.device_count()
            print(f"Using all available GPUs: {world_size}")
        
        if world_size < 2:
            print("Warning: Less than 2 GPUs available, falling back to single GPU")
            # Fall back to single GPU training
            args.use_multi_gpu = False
        else:
            print(f"Starting DDP training with {world_size} GPUs")
            # Launch DDP processes
            mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
            return
    
    # Single GPU training (fallback)
    print("Using single GPU training")
    train_worker(0, 1, args)

if __name__ == "__main__":
    main() 