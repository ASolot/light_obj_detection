#!/bin/bash
#!
#! Example SLURM job script for Wilkes2 (Broadwell, ConnectX-4, P100)
#! Last updated: Mon 13 Nov 12:06:57 GMT 2017
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J ams288-yolov3-infer
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A MASCOLO-SL3-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks=1
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#! Note that the job submission script will enforce no more than 3 cpus per GPU.
#SBATCH --gres=gpu:1
#! How much wallclock time will be required?
#SBATCH --time=02:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! Do not change:
#SBATCH -p pascal

#! sbatch directives end here (put any additional directives above this line)

#! Notes:
#! Charging is determined by GPU number*walltime. 

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
#! ############################################################
#! Modify the settings below to specify the application's environment, location 
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel7/default-gpu              # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:

# conda environment
module load miniconda3-4.5.4-gcc-5.4.0-hivczbz

# # CUDA 10.0 & cudnn
# module load cuda/10.0 intel/mkl/2017.4 cudnn/7.5_cuda-10.0

module unload cuda/8.0
# for unknown reasons, tensorboard won't work 
module load cuda/10.0

# CUDA 9.0 & cudnn 
module load cuda/9.0 intel/mkl/2017.4 
module load cudnn/7.3_cuda-9.0


#! Activate the virtual environment specific for the task
source activate thesisenvcl

#! Full path to application executable: 
application="$HOME/MThesis/repos/mine/light_obj_detection/SolotNet/models/backbones/DCNv2/make.sh"

#! Run options for the application:
options=""

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 12:
export OMP_NUM_THREADS=1

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):
CMD="$application $options"

#! Choose this for a MPI code using OpenMPI:
#CMD="mpirun -npernode $mpi_tasks_per_node -np $np $application $options"


###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD 
