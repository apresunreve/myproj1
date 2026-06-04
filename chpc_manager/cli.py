#!/usr/bin/env python3
"""
Command-line interface for CHPC Job Manager
Provides commands to add tasks, monitor status, and manage the system
"""
import argparse
import sys
import json
import os
from typing import Dict, Any

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


def _print_table(headers, rows):
    if tabulate is not None:
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        return

    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt(row):
        return " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)
    print(fmt(headers))
    print(sep)
    for row in rows:
        print(fmt(row))

from task_db import TaskDB
from config_manager import Config
from slurm_interface import SlurmInterface


def add_task(args):
    """Add a new task to the database"""
    config = Config(args.config)
    db = TaskDB(config.database_path)
    
    # Parse arguments JSON
    try:
        arguments = json.loads(args.arguments)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON for arguments: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Get completion file path if configured
    try:
        completion_file = config.get_completion_file(args.task_type, arguments)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Insert task
    task_id = db.insert_task(
        task_type=args.task_type,
        arguments=arguments,
        partition=args.partition,
        completion_file=completion_file
    )
    
    print(f"Task {task_id} added successfully")
    print(f"  Type: {args.task_type}")
    print(f"  Arguments: {arguments}")
    if args.partition:
        print(f"  Partition: {args.partition}")
    if completion_file:
        print(f"  Completion file: {completion_file}")
    
    db.close()


def list_tasks(args):
    """List tasks with optional filtering"""
    config = Config(args.config)
    db = TaskDB(config.database_path)
    
    # Get tasks based on filter
    if args.status:
        if args.status == 'running':
            tasks = db.get_running_tasks()
        else:
            tasks = db.get_tasks_by_status(args.status)
    else:
        tasks = db.get_all_tasks()
    
    if not tasks:
        print("No tasks found")
        db.close()
        return
    
    # Prepare table data
    table_data = []
    for task in tasks:
        # Simplify status for display
        status = task['status']
        if status.startswith('running:'):
            display_status = 'running'
            slurm_id = status.split(':', 1)[1]
        else:
            display_status = status
            slurm_id = '-'
        
        args_str = json.dumps(task['arguments'])
        if len(args_str) > 50:
            args_str = args_str[:47] + '...'
        
        table_data.append([
            task['id'],
            task['task_type'],
            args_str,
            display_status,
            slurm_id,
            task['partition'] or '-',
            task['created_at']
        ])
    
    # Print table
    headers = ['ID', 'Type', 'Arguments', 'Status', 'SLURM Job', 'Partition', 'Created']
    _print_table(headers, table_data)
    
    db.close()


def status(args):
    """Show overall system status"""
    config = Config(args.config)
    db = TaskDB(config.database_path)
    # Extract cluster names from partition configs
    clusters = list(set(p.get('cluster') for p in config.partitions if p.get('cluster')))
    slurm = SlurmInterface(config.username, clusters=clusters)
    
    # Get task counts
    idle = db.count_idle_tasks()
    running = db.count_running_tasks()
    done = db.count_done_tasks()
    total = idle + running + done
    
    print("=== CHPC Job Manager Status ===\n")
    print(f"Total tasks: {total}")
    print(f"  Idle:    {idle}")
    print(f"  Running: {running}")
    print(f"  Done:    {done}")
    
    # Get SLURM job information
    print("\n=== SLURM Jobs ===\n")
    
    # Filter by task-executor jobs to match what manager daemon tracks
    job_name_prefix = "task-executor-"
    pending_by_partition = slurm.get_pending_jobs_by_partition(job_name_prefix=job_name_prefix)
    all_jobs = slurm.get_user_jobs()
    
    if not all_jobs:
        print("No active SLURM jobs")
    else:
        job_table = []
        for job in all_jobs:
            job_table.append([
                job.job_id,
                job.name,
                job.partition,
                job.state,
                job.time,
                job.nodelist
            ])
        
        headers = ['Job ID', 'Name', 'Partition', 'State', 'Time', 'Nodes']
        _print_table(headers, job_table)
    
    # Show partition status
    print("\n=== Partition Status ===\n")
    partition_table = []
    for partition_config in config.partitions:
        partition_name = partition_config['name']
        target = partition_config['target_pending_jobs']
        current = pending_by_partition.get(partition_name, 0)
        
        partition_table.append([
            partition_name,
            current,
            target,
            partition_config['max_walltime']
        ])
    
    headers = ['Partition', 'Pending', 'Target', 'Max Walltime']
    _print_table(headers, partition_table)
    
    db.close()


def task_details(args):
    """Show detailed information about a specific task"""
    config = Config(args.config)
    db = TaskDB(config.database_path)
    
    task = db.get_task(args.task_id)
    
    if not task:
        print(f"Task {args.task_id} not found", file=sys.stderr)
        db.close()
        sys.exit(1)
    
    print(f"=== Task {task['id']} ===\n")
    print(f"Type:            {task['task_type']}")
    print(f"Status:          {task['status']}")
    print(f"SLURM Job ID:    {task['slurm_job_id'] or 'N/A'}")
    print(f"Partition:       {task['partition'] or 'N/A'}")
    print(f"Created:         {task['created_at']}")
    print(f"Updated:         {task['updated_at']}")
    print(f"Completion file: {task['completion_file'] or 'N/A'}")
    print(f"\nArguments:")
    print(json.dumps(task['arguments'], indent=2))
    
    db.close()


def check_completion(args):
    """Check completion file status for tasks and optionally update status"""
    config = Config(args.config)
    db = TaskDB(config.database_path)

    updated = 0
    for task_id in args.id:
        task = db.get_task(task_id)
        if not task:
            print(f"Task {task_id} not found", file=sys.stderr)
            continue

        completion_file = task.get('completion_file')
        exists = bool(completion_file and os.path.exists(completion_file))
        print(f"Task {task_id}: completion_file={completion_file or 'N/A'} exists={exists}")

        if exists and args.mark_done:
            db.update_task_status(task_id, 'done')
            updated += 1
            print(f"  Updated status -> done")
        elif (not exists) and args.mark_idle:
            db.update_task_status(task_id, 'idle')
            updated += 1
            print(f"  Updated status -> idle")

    if updated:
        print(f"Updated {updated} task(s)")
    db.close()


def set_status(args):
    """Manually set task status"""
    config = Config(args.config)
    db = TaskDB(config.database_path)

    for task_id in args.id:
        task = db.get_task(task_id)
        if not task:
            print(f"Task {task_id} not found", file=sys.stderr)
            continue

        if args.status == 'running':
            slurm_id = args.slurm_job_id or 'manual'
            status = f"running:{slurm_id}"
            db.update_task_status(task_id, status, slurm_id)
        else:
            db.update_task_status(task_id, args.status)

        print(f"Task {task_id} status -> {args.status}")

    db.close()


def set_done(args):
    """Manually set task status to done by task id"""
    config = Config(args.config)
    db = TaskDB(config.database_path)

    updated = 0
    for task_id in args.id:
        task = db.get_task(task_id)
        if not task:
            print(f"Task {task_id} not found", file=sys.stderr)
            continue

        db.update_task_status(task_id, 'done')
        updated += 1
        print(f"Task {task_id} status -> done")

    if updated:
        print(f"Updated {updated} task(s)")
    db.close()


def set_done_by_job_id(args):
    """Manually set task status to done by SLURM job id"""
    config = Config(args.config)
    db = TaskDB(config.database_path)

    updated = 0
    for job_id in args.job_id:
        tasks = db.get_tasks_by_slurm_job_id(job_id)
        if not tasks:
            print(f"No tasks found for SLURM job id {job_id}", file=sys.stderr)
            continue

        for task in tasks:
            db.update_task_status(task['id'], 'done')
            updated += 1
            print(f"Task {task['id']} (job {job_id}) status -> done")

    if updated:
        print(f"Updated {updated} task(s)")
    db.close()


def clear_done(args):
    """Clear completed tasks from database"""
    config = Config(args.config)
    db = TaskDB(config.database_path)
    
    done_tasks = db.get_tasks_by_status('done')
    count = len(done_tasks)
    
    if count == 0:
        print("No completed tasks to clear")
        db.close()
        return
    
    if not args.force:
        response = input(f"Clear {count} completed tasks? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled")
            db.close()
            return
    
    # Delete completed tasks
    conn = db._get_connection()
    conn.execute("DELETE FROM tasks WHERE status = 'done'")
    conn.commit()
    
    print(f"Cleared {count} completed tasks")
    db.close()


def delete_tasks(args):
    """Delete tasks by id, status, or task_type. Requires confirmation unless --force."""
    config = Config(args.config)
    db = TaskDB(config.database_path)

    # Build WHERE clause from filters
    where_clauses = []
    params = []

    if args.id:
        placeholders = ','.join(['?'] * len(args.id))
        where_clauses.append(f"id IN ({placeholders})")
        params.extend(args.id)

    if args.status:
        if args.status == 'running':
            where_clauses.append("status LIKE 'running:%'")
        else:
            where_clauses.append("status = ?")
            params.append(args.status)

    if args.task_type:
        where_clauses.append("task_type = ?")
        params.append(args.task_type)

    where_sql = ''
    if where_clauses:
        where_sql = 'WHERE ' + ' AND '.join(where_clauses)
    else:
        # No filters -- dangerous operation
        if not args.force:
            resp = input("No filters specified. This will delete ALL tasks. Continue? [y/N]: ")
            if resp.lower() != 'y':
                print("Cancelled")
                db.close()
                return

    # Count matching tasks
    conn = db._get_connection()
    try:
        # If dry-run, fetch matching rows to display
        if args.dry_run:
            cur = conn.execute(f"SELECT id, task_type, arguments, status, slurm_job_id, partition, created_at FROM tasks {where_sql} ORDER BY created_at DESC", tuple(params))
            rows = cur.fetchall()
            if not rows:
                print("No matching tasks found")
                db.close()
                return

            print(f"Dry-run: {len(rows)} matching task(s)")
            for row in rows:
                try:
                    args_json = json.loads(row['arguments'])
                    args_str = json.dumps(args_json)
                except Exception:
                    args_str = row['arguments']
                slurm_id = row['slurm_job_id'] or '-'
                print(f"ID={row['id']} Type={row['task_type']} Status={row['status']} SLURM={slurm_id} Partition={row['partition'] or '-'} Created={row['created_at']} Args={args_str}")
            db.close()
            return

        cur = conn.execute(f"SELECT COUNT(*) as cnt FROM tasks {where_sql}", tuple(params))
        count = cur.fetchone()['cnt']
    except Exception as e:
        print(f"Error counting tasks: {e}")
        db.close()
        return

    if count == 0:
        print("No matching tasks found")
        db.close()
        return

    print(f"Found {count} matching task(s)")
    if not args.force:
        resp = input("Proceed with deletion? [y/N]: ")
        if resp.lower() != 'y':
            print("Cancelled")
            db.close()
            return

    # Perform deletion
    try:
        conn.execute(f"DELETE FROM tasks {where_sql}", tuple(params))
        conn.commit()
        print(f"Deleted {count} task(s)")
    except Exception as e:
        print(f"Error deleting tasks: {e}")
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="CHPC Job Manager CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add a task
  %(prog)s add nas_training '{"experiment_id": "exp001", "dataset_percent": 0.1}'
  
  # List all tasks
  %(prog)s list
  
  # List only idle tasks
  %(prog)s list --status idle
  
  # Show system status
  %(prog)s status
  
  # Show task details
  %(prog)s task 42
  
  # Check completion file for task(s) and mark done/idle
  %(prog)s check --id 42 --mark-done
  
  # Force status (e.g., reset a stuck task)
  %(prog)s set-status --id 42 --status idle

  # Mark task(s) done by task id
  %(prog)s set-done --id 42 43

  # Mark task(s) done by SLURM job id
  %(prog)s set-done-job --job-id 123456
  
  # Clear completed tasks
  %(prog)s clear-done
        """
    )
    
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Add task command
    add_parser = subparsers.add_parser('add', help='Add a new task')
    add_parser.add_argument('task_type', help='Task type (must be defined in task_types.yaml)')
    add_parser.add_argument('arguments', help='Task arguments as JSON string')
    add_parser.add_argument('--partition', help='Preferred partition (optional)')
    add_parser.set_defaults(func=add_task)
    
    # List tasks command
    list_parser = subparsers.add_parser('list', help='List tasks')
    list_parser.add_argument('--status', choices=['idle', 'running', 'done'],
                            help='Filter by status')
    list_parser.set_defaults(func=list_tasks)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    status_parser.set_defaults(func=status)
    
    # Task details command
    task_parser = subparsers.add_parser('task', help='Show task details')
    task_parser.add_argument('task_id', type=int, help='Task ID')
    task_parser.set_defaults(func=task_details)

    # Check completion command
    check_parser = subparsers.add_parser('check', help='Check completion file for task(s)')
    check_parser.add_argument('--id', type=int, nargs='+', required=True, help='One or more task IDs to check')
    check_parser.add_argument('--mark-done', action='store_true', help='Mark done if completion file exists')
    check_parser.add_argument('--mark-idle', action='store_true', help='Mark idle if completion file missing')
    check_parser.set_defaults(func=check_completion)

    # Set status command
    set_parser = subparsers.add_parser('set-status', help='Manually set task status')
    set_parser.add_argument('--id', type=int, nargs='+', required=True, help='One or more task IDs to update')
    set_parser.add_argument('--status', choices=['idle', 'running', 'done'], required=True)
    set_parser.add_argument('--slurm-job-id', help='SLURM job id for running status (optional)')
    set_parser.set_defaults(func=set_status)

    # Set done command
    done_parser = subparsers.add_parser('set-done', help='Mark task(s) done by task id')
    done_parser.add_argument('--id', type=int, nargs='+', required=True, help='One or more task IDs to update')
    done_parser.set_defaults(func=set_done)

    # Set done by SLURM job id command
    set_done_parser = subparsers.add_parser('set-done-job', help='Mark task(s) done by SLURM job id')
    set_done_parser.add_argument('--job-id', nargs='+', required=True, help='One or more SLURM job IDs to update')
    set_done_parser.set_defaults(func=set_done_by_job_id)
    
    # Clear done command
    clear_parser = subparsers.add_parser('clear-done', help='Clear completed tasks')
    clear_parser.add_argument('--force', action='store_true', 
                             help='Do not ask for confirmation')
    clear_parser.set_defaults(func=clear_done)

    # Delete tasks command
    del_parser = subparsers.add_parser('delete', help='Delete tasks by id, status, or type')
    del_parser.add_argument('--id', type=int, nargs='+', help='One or more task IDs to delete')
    del_parser.add_argument('--status', choices=['idle', 'running', 'done'], help='Delete tasks by status')
    del_parser.add_argument('--task-type', dest='task_type', help='Delete tasks by task_type')
    del_parser.add_argument('--force', action='store_true', help='Do not ask for confirmation')
    del_parser.add_argument('--dry-run', action='store_true', help='Show matching tasks without deleting them')
    del_parser.set_defaults(func=delete_tasks)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Run command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
