#!/usr/bin/env python3
"""
Manager Daemon - Combines Job Launcher (JL) and Restart Daemon (RD)
Periodically checks task queue and SLURM jobs, manages task-executor submissions
"""
import os
import sys
import time
import signal
import logging
import argparse
from datetime import datetime
from typing import Set

from task_db import TaskDB
from slurm_interface import SlurmInterface
from config_manager import Config
from gpu_utils import get_nodes_with_insufficient_memory, merge_exclusion_lists


class ManagerDaemon:
    """
    Manager daemon that combines Restart Daemon and Job Launcher functionality
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = Config(config_path)
        self.db = TaskDB(self.config.database_path)
        # Extract cluster names from partition configs
        clusters = list(set(p.get('cluster') for p in self.config.partitions if p.get('cluster')))
        self.slurm = SlurmInterface(self.config.username, clusters=clusters)
        self.running = True
        self.logger = self._setup_logging()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for manager daemon"""
        log_file = self.config.manager_settings.get('log_file', './manager.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        return logging.getLogger(__name__)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self.running = False
    
    def restart_daemon(self) -> bool:
        """
        Restart Daemon: Check for incomplete tasks and reset them
        Returns True if any tasks were reset (triggers job launcher)
        """
        self.logger.info("Running Restart Daemon check...")
        
        # Get all running job IDs from SLURM
        active_job_ids = set(self.slurm.get_running_job_ids())
        self.logger.debug(f"Active SLURM jobs: {active_job_ids}")
        
        # Get all tasks marked as running in database
        running_tasks = self.db.get_running_tasks()
        
        tasks_reset = 0
        tasks_completed = 0
        
        for task in running_tasks:
            task_id = task['id']
            slurm_job_id = task['slurm_job_id']
            completion_file = task['completion_file']
            
            # Check if completion file exists
            if completion_file and os.path.exists(completion_file):
                self.logger.info(
                    f"Task {task_id} completed (completion file found). Marking as done."
                )
                self.db.update_task_status(task_id, 'done')
                tasks_completed += 1
                continue
            
            # Check if job is still in SLURM queue
            if slurm_job_id not in active_job_ids:
                # Job disappeared (preempted, timed out, or crashed)
                self.logger.warning(
                    f"Task {task_id} job {slurm_job_id} not in queue. Resetting to idle."
                )
                self.db.update_task_status(task_id, 'idle')
                tasks_reset += 1
        
        if tasks_reset > 0:
            self.logger.info(f"Reset {tasks_reset} incomplete tasks to idle")
        if tasks_completed > 0:
            self.logger.info(f"Marked {tasks_completed} tasks as completed")
        
        return tasks_reset > 0
    
    def job_launcher(self):
        """
        Job Launcher: Submit task-executor jobs to maintain target pending jobs
        """
        self.logger.info("Running Job Launcher...")
        
        # Check if there are idle tasks
        idle_count = self.db.count_idle_tasks()
        
        if idle_count == 0:
            self.logger.info("No idle tasks. Skipping job submission.")
            return
        
        self.logger.info(f"Found {idle_count} idle tasks")
        
        # Submit jobs for each configured partition
        for partition_config in self.config.partitions:
            partition_name = partition_config['name']
            target_pending = partition_config['target_pending_jobs']
            max_walltime = partition_config['max_walltime']
            
            # Count current pending task-executor jobs for this partition
            # Only count jobs with name "task-executor-{partition}"
            job_name_prefix = "NAS2-"
            # job_name_prefix = "task-executor-"
            current_pending = self.slurm.count_pending_jobs(
                partition=partition_name,
                job_name_prefix=job_name_prefix
            )
            
            self.logger.info(
                f"Partition {partition_name}: {current_pending}/{target_pending} pending jobs"
            )
            
            # Calculate how many jobs to submit
            jobs_to_submit = target_pending - current_pending
            
            if jobs_to_submit > 0:
                self._submit_task_executor_jobs(
                    partition_config,
                    jobs_to_submit
                )
    
    def _submit_task_executor_jobs(self, partition_config: dict, count: int):
        """Submit task-executor jobs to a partition"""
        partition = partition_config['name']
        walltime = partition_config['max_walltime']
        cluster = partition_config.get('cluster')
        account = partition_config.get('account')
        qos = partition_config.get('qos')
        partition_exclude = partition_config.get('exclude_nodes', [])
        
        # Start with partition-level exclusions
        exclude_nodes = list(partition_exclude) if partition_exclude else []
        
        # Gather resource requirements from idle tasks
        # This includes GPU requests (gres) and GPU memory filtering
        resources = {}
        if cluster and self.db.count_idle_tasks() > 0:
            # Get all unique task types from idle tasks
            idle_tasks = self.db.get_tasks_by_status('idle')
            task_types = set(task['task_type'] for task in idle_tasks)
            
            # Collect GPU memory requirements and ALL resource specs from all task types
            gpu_memory_exclusions = []
            max_gres = None  # Track maximum GPU requirement
            max_mem = None  # Track maximum memory requirement
            max_cpus = None  # Track maximum CPU requirement
            max_nodes = None  # Track maximum node requirement
            
            for task_type in task_types:
                try:
                    task_config = self.config.get_task_type(task_type)
                    default_resources = task_config.get('default_resources', {})
                    
                    # GPU memory filtering
                    min_gpu_memory = default_resources.get('gpu_memory_gb')
                    if min_gpu_memory:
                        min_gpu_memory = int(min_gpu_memory)
                        # Get nodes with insufficient memory for this cluster
                        insufficient_nodes = get_nodes_with_insufficient_memory(
                            cluster, min_gpu_memory
                        )
                        gpu_memory_exclusions.extend(insufficient_nodes)
                        
                        if insufficient_nodes:
                            self.logger.debug(
                                f"Task type {task_type} requires {min_gpu_memory}GB GPU memory, "
                                f"excluding {len(insufficient_nodes)} nodes"
                            )
                    
                    # Collect GPU resource requirements (gres)
                    # Use the maximum GPU request among all task types
                    if 'gres' in default_resources:
                        current_gres = default_resources['gres']
                        if max_gres is None:
                            max_gres = current_gres
                        else:
                            # Parse GPU count from gres strings like "gpu:2" or "gpu:a100:1"
                            # Use the one with more GPUs
                            try:
                                current_count = int(current_gres.split(':')[-1])
                                max_count = int(max_gres.split(':')[-1])
                                if current_count > max_count:
                                    max_gres = current_gres
                            except (ValueError, IndexError):
                                # If parsing fails, keep the first one
                                pass
                    
                    # Collect memory requirements - use maximum
                    if 'mem' in default_resources:
                        current_mem = default_resources['mem']
                        if max_mem is None:
                            max_mem = current_mem
                        else:
                            # Parse memory values like "16G", "32GB", "2000M"
                            # For simplicity, use the one from the task with most GPUs
                            # or just use the larger value
                            try:
                                def parse_mem(mem_str):
                                    mem_str = mem_str.upper().replace('B', '')
                                    if 'G' in mem_str:
                                        return float(mem_str.replace('G', '')) * 1024
                                    elif 'M' in mem_str:
                                        return float(mem_str.replace('M', ''))
                                    return float(mem_str)
                                
                                if parse_mem(current_mem) > parse_mem(max_mem):
                                    max_mem = current_mem
                            except (ValueError, AttributeError):
                                pass
                    
                    # Collect CPU requirements - use maximum
                    if 'cpus_per_task' in default_resources:
                        current_cpus = int(default_resources['cpus_per_task'])
                        if max_cpus is None or current_cpus > max_cpus:
                            max_cpus = current_cpus
                    
                    # Collect node requirements - use maximum
                    if 'nodes' in default_resources:
                        current_nodes = int(default_resources['nodes'])
                        if max_nodes is None or current_nodes > max_nodes:
                            max_nodes = current_nodes
                
                except Exception as e:
                    self.logger.warning(
                        f"Could not process requirements for {task_type}: {e}"
                    )
            
            # Set resource requests for TE jobs
            resource_log_parts = []
            if max_gres:
                resources['gres'] = max_gres
                resource_log_parts.append(f"gres={max_gres}")
            if max_mem:
                resources['mem'] = max_mem
                resource_log_parts.append(f"mem={max_mem}")
            if max_cpus:
                resources['cpus_per_task'] = max_cpus
                resource_log_parts.append(f"cpus={max_cpus}")
            if max_nodes:
                resources['nodes'] = max_nodes
                resource_log_parts.append(f"nodes={max_nodes}")
            
            if resource_log_parts:
                self.logger.debug(f"Task-executor jobs will request: {', '.join(resource_log_parts)}")
            
            # Merge all exclusion lists
            if gpu_memory_exclusions:
                exclude_nodes = merge_exclusion_lists(exclude_nodes, gpu_memory_exclusions)
                self.logger.debug(
                    f"Total nodes excluded for {cluster}/{partition}: {len(exclude_nodes)}"
                )
        
        project_dir = os.path.dirname(os.path.abspath(__file__))
        executor_script = os.path.join(project_dir, 'task_executor_wrapper.sh')
        
        # Create wrapper script if it doesn't exist
        self._create_executor_wrapper(executor_script)
        
        for i in range(count):
            try:
                job_name = f"NAS2-{partition}"
                # job_name = f"task-executor-{partition}"
                
                # Get log paths
                log_dir = self.config.task_executor_settings.get('log_dir', './task_logs')
                os.makedirs(log_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_log = os.path.join(log_dir, f"executor_{partition}_{timestamp}_%j.out")
                error_log = os.path.join(log_dir, f"executor_{partition}_{timestamp}_%j.err")
                
                # Submit job with CHPC-specific parameters and resource requirements
                job_id = self.slurm.submit_job(
                    script_path=executor_script,
                    job_name=job_name,
                    partition=partition,
                    walltime=walltime,
                    output_log=output_log,
                    error_log=error_log,
                    cluster=cluster,
                    account=account,
                    qos=qos,
                    exclude_nodes=exclude_nodes if exclude_nodes else None,
                    resources=resources if resources else None
                )
                
                # Log submission details
                log_parts = [f"Submitted task-executor job {job_id} to partition {partition}"]
                if resources and 'gres' in resources:
                    log_parts.append(f"with {resources['gres']}")
                if exclude_nodes:
                    log_parts.append(f"(excluding {len(exclude_nodes)} nodes)")
                self.logger.info(' '.join(log_parts))
            
            except Exception as e:
                self.logger.error(
                    f"Failed to submit task-executor job to {partition}: {e}"
                )
    
    def _create_executor_wrapper(self, wrapper_path: str):
        """Create SLURM wrapper script for task executor"""
        project_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(project_dir, 'config.yaml')
        executor_path = os.path.join(project_dir, 'task_executor.py')
        
        wrapper_content = f"""#!/bin/bash
# Note: job name is set via sbatch --job-name parameter, not here

# Get the SLURM job ID
SLURM_JOB_ID=${{SLURM_JOB_ID:-"unknown"}}

# Change to project directory
cd {project_dir}

# Run the task executor
python3 {executor_path} --config {config_path} --slurm-job-id $SLURM_JOB_ID
"""
        
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_content)
        
        # Make executable
        os.chmod(wrapper_path, 0o755)
    
    def run(self):
        """Main daemon loop"""
        self.logger.info("Manager Daemon started")
        
        check_interval = self.config.manager_settings.get('check_interval', 300)
        self.logger.info(f"Check interval: {check_interval} seconds")
        
        while self.running:
            try:
                # Run Restart Daemon
                tasks_were_reset = self.restart_daemon()
                
                # Run Job Launcher
                # Always run it, but it's especially important if tasks were reset
                if tasks_were_reset:
                    self.logger.info("Tasks were reset - running job launcher")
                
                self.job_launcher()
                
                # Print status summary
                idle = self.db.count_idle_tasks()
                running = self.db.count_running_tasks()
                done = self.db.count_done_tasks()
                
                self.logger.info(
                    f"Status: {idle} idle, {running} running, {done} done"
                )
                
                # Sleep until next check
                if self.running:
                    self.logger.info(f"Sleeping for {check_interval} seconds...")
                    time.sleep(check_interval)
            
            except Exception as e:
                self.logger.error(f"Error in daemon loop: {e}", exc_info=True)
                # Sleep a bit before retrying
                time.sleep(60)
        
        self.logger.info("Manager Daemon stopped")
        self.db.close()


def main():
    parser = argparse.ArgumentParser(description="CHPC Job Manager Daemon")
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    daemon = ManagerDaemon(args.config)
    daemon.run()


if __name__ == '__main__':
    main()
