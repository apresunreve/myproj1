"""
SLURM interface for job submission and monitoring
"""
import subprocess
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class SlurmJob:
    """Represents a SLURM job"""
    job_id: str
    partition: str
    name: str
    user: str
    state: str  # PD, R, CG, etc.
    time: str
    nodes: int
    nodelist: str


class SlurmInterface:
    """Wrapper for SLURM commands"""
    
    def __init__(self, username: str, clusters: Optional[List[str]] = None):
        self.username = username
        self.clusters = clusters or []  # List of cluster names to query
    
    def submit_job(self, script_path: str, job_name: str, partition: str,
                   walltime: str, resources: Optional[Dict[str, Any]] = None,
                   output_log: Optional[str] = None, 
                   error_log: Optional[str] = None,
                   cluster: Optional[str] = None,
                   account: Optional[str] = None,
                   qos: Optional[str] = None,
                   exclude_nodes: Optional[List[str]] = None) -> str:
        """
        Submit a job using sbatch
        Returns the SLURM job ID
        """
        cmd = ['sbatch']
        
        # CHPC-specific parameters
        if cluster:
            cmd.extend(['-M', cluster])
        if account:
            cmd.extend(['--account', account])
        if qos:
            cmd.extend(['--qos', qos])
        
        # Basic parameters
        cmd.extend(['--job-name', job_name])
        cmd.extend(['--partition', partition])
        cmd.extend(['--time', walltime])
        
        # Node exclusion
        if exclude_nodes:
            nodes_str = ','.join(exclude_nodes)
            cmd.extend(['--exclude', nodes_str])
        
        # Output/error logs
        if output_log:
            cmd.extend(['--output', output_log])
        if error_log:
            cmd.extend(['--error', error_log])
        
        # Resource requirements
        if resources:
            if 'nodes' in resources:
                cmd.extend(['--nodes', str(resources['nodes'])])
            if 'ntasks' in resources:
                cmd.extend(['--ntasks', str(resources['ntasks'])])
            if 'cpus_per_task' in resources:
                cmd.extend(['--cpus-per-task', str(resources['cpus_per_task'])])
            if 'mem' in resources:
                cmd.extend(['--mem', resources['mem']])
            if 'gres' in resources:
                cmd.extend(['--gres', resources['gres']])
        
        # Script to run
        cmd.append(script_path)
        
        # Execute sbatch
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse job ID from output: "Submitted batch job 12345"
            match = re.search(r'Submitted batch job (\d+)', result.stdout)
            if match:
                return match.group(1)
            else:
                raise RuntimeError(f"Could not parse job ID from: {result.stdout}")
        
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to submit job: {e.stderr}"
            )
    
    def get_user_jobs(self, states: Optional[List[str]] = None) -> List[SlurmJob]:
        """
        Get all jobs for the current user across all configured clusters
        Optionally filter by states (e.g., ['PD', 'R'])
        """
        all_jobs = []
        
        # If clusters configured, query each one
        if self.clusters:
            for cluster in self.clusters:
                all_jobs.extend(self._get_jobs_for_cluster(cluster, states))
        else:
            # No clusters configured, query default
            all_jobs = self._get_jobs_for_cluster(None, states)
        
        return all_jobs
    
    def _get_jobs_for_cluster(self, cluster: Optional[str], states: Optional[List[str]] = None) -> List[SlurmJob]:
        """
        Get jobs for a specific cluster (or default if cluster is None)
        """
        # Use %.100P and %.100j to avoid truncation of partition and job names
        cmd = ['squeue', '-u', self.username, '--noheader', '--format=%i|%.100P|%.100j|%u|%T|%M|%D|%R']
        
        if cluster:
            cmd.extend(['-M', cluster])
        
        if states:
            state_filter = ','.join(states)
            cmd.extend(['-t', state_filter])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            jobs = []
            for line in result.stdout.strip().split('\n'):
                if not line or line.startswith('CLUSTER:'):
                    continue
                
                parts = line.split('|')
                if len(parts) == 8:
                    # Strip asterisk from partition name (SLURM adds * to default partition)
                    partition_name = parts[1].strip().rstrip('*')
                    jobs.append(SlurmJob(
                        job_id=parts[0].strip(),
                        partition=partition_name,
                        name=parts[2].strip(),
                        user=parts[3].strip(),
                        state=parts[4].strip(),
                        time=parts[5].strip(),
                        nodes=int(parts[6].strip()) if parts[6].strip().isdigit() else 0,
                        nodelist=parts[7].strip()
                    ))
            
            return jobs
        
        except subprocess.CalledProcessError as e:
            # If squeue fails for this cluster, return empty list
            return []
    
    def get_running_job_ids(self) -> List[str]:
        """Get list of all running job IDs for the user"""
        jobs = self.get_user_jobs(states=['RUNNING', 'R'])
        return [job.job_id for job in jobs]
    
    def get_pending_jobs_by_partition(self, job_name_prefix: str = None) -> Dict[str, int]:
        """
        Get count of pending jobs per partition
        Optionally filter by job name prefix (e.g., "task-executor-")
        Returns dict: {partition_name: pending_count}
        """
        jobs = self.get_user_jobs(states=['PENDING', 'PD'])
        
        pending_by_partition = {}
        for job in jobs:
            # Optionally filter by job name prefix
            if job_name_prefix is not None and not job.name.startswith(job_name_prefix):
                continue
            partition = job.partition
            pending_by_partition[partition] = pending_by_partition.get(partition, 0) + 1
        
        return pending_by_partition
    
    def count_pending_jobs(self, partition: str, job_name_prefix: str = None) -> int:
        """
        Count pending jobs for a specific partition
        Optionally filter by job name prefix (e.g., "task-executor-")
        """
        jobs = self.get_user_jobs(states=['PENDING', 'PD'])
        
        count = 0
        for job in jobs:
            if job.partition == partition:
                if job_name_prefix is None or job.name.startswith(job_name_prefix):
                    count += 1
        
        return count
    
    def cancel_job(self, job_id: str):
        """Cancel a job using scancel"""
        try:
            subprocess.run(
                ['scancel', job_id],
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to cancel job {job_id}: {e.stderr}")
    
    def job_exists(self, job_id: str) -> bool:
        """Check if a job ID exists in the queue"""
        all_jobs = self.get_user_jobs()
        return any(job.job_id == job_id for job in all_jobs)
