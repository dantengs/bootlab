#!/bin/bash
# bootlab's wrapper for one node's MPI program (execute.srun_argv when a step runs on one node): a one-task srun
# step runs it on the host, then it execs the rest of the command line, `apptainer exec <image> mpirun -np <n>
# ... <program>`: the image's own mpirun starts every rank in that one container, on the step's cores. ob1 over
# vader (shared memory) carries all traffic and no network device is opened: 256 ranks on a 256-core node start in
# 0.4 s and end in 0.1 s, against 21 s and 8.7 s with srun --mpi=pmix and UCX (measured on
# the development cluster). Slurm's and PMIx's variables go: with them mpirun would take the step's one task as its one slot;
# without them its slots are the cores of the step's cpuset. On SIGTERM (scancel, the time limit) mpirun signals
# the ranks, which write their checkpoint, and waits 25 s before SIGKILL (Open MPI's default is 1 s; Slurm's
# KillWait on the development cluster is 30 s).
for v in ${!SLURM_@} ${!PMIX_@} ${!PMI_@}; do unset "$v"; done
export OMPI_MCA_plm=isolated OMPI_MCA_pml=ob1 OMPI_MCA_btl=vader,self OMPI_MCA_odls_base_sigkill_timeout=25
exec "$@"
