#!/bin/bash
#SBATCH --job-name="adk_explicit_MD"
#SBATCH --output="adk_explicit_MD_%j.out"                        
#SBATCH --partition=gpu                                        
#SBATCH --nodes=1                                              
#SBATCH --gpus-per-node=4                                     
#SBATCH --ntasks-per-node=4                                    
#SBATCH --cpus-per-task=16                                      
#SBATCH --mem=256000                                           
#SBATCH --time=167:59:59                                        
                        
module load openmm
module load cuda
source activate SEEKR2

export CUDA_VISIBLE_DEVICES=0
cd /mnt/home/aojha/ceph/ensemble_reweighting/adk_iterative_WE/adk_explicit_MD/init_MD
python simulation.py -c 0


