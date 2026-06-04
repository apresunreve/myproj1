"""
Task Database management using SQLite with atomic operations
"""
import sqlite3
import json
import threading
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
from datetime import datetime


class TaskDB:
    """
    Thread-safe Task Database using SQLite
    Handles atomic operations for task insertion, extraction, and status updates
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()
    
    def _get_connection(self):
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path,
                isolation_level='IMMEDIATE',  # Ensure atomicity
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    @contextmanager
    def _transaction(self):
        """Context manager for atomic database transactions"""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
    
    def _init_db(self):
        """Initialize database schema"""
        with self._transaction() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_type TEXT NOT NULL,
                    arguments TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'idle',
                    slurm_job_id TEXT,
                    partition TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completion_file TEXT
                )
            """)
            
            # Create indexes for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON tasks(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_type ON tasks(task_type)
            """)
    
    def insert_task(self, task_type: str, arguments: Dict[str, Any], 
                    partition: Optional[str] = None,
                    completion_file: Optional[str] = None) -> int:
        """
        Insert a new task into the database
        Returns the task ID
        """
        now = datetime.now().isoformat()
        arguments_json = json.dumps(arguments)
        
        with self._transaction() as conn:
            cursor = conn.execute("""
                INSERT INTO tasks (task_type, arguments, status, partition, 
                                   created_at, updated_at, completion_file)
                VALUES (?, ?, 'idle', ?, ?, ?, ?)
            """, (task_type, arguments_json, partition, now, now, completion_file))
            
            return cursor.lastrowid
    
    def get_next_idle_task(self) -> Optional[Dict[str, Any]]:
        """
        Atomically get and reserve the next idle task for execution
        Returns None if no idle tasks available
        """
        with self._transaction() as conn:
            # Find an idle task
            cursor = conn.execute("""
                SELECT id, task_type, arguments, partition, completion_file
                FROM tasks
                WHERE status = 'idle'
                ORDER BY created_at ASC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Return task data
            return {
                'id': row['id'],
                'task_type': row['task_type'],
                'arguments': json.loads(row['arguments']),
                'partition': row['partition'],
                'completion_file': row['completion_file']
            }
    
    def update_task_status(self, task_id: int, status: str, 
                          slurm_job_id: Optional[str] = None):
        """
        Update the status of a task
        Status can be: 'idle', 'running:<slurm_job_id>', 'done'
        """
        now = datetime.now().isoformat()
        
        with self._transaction() as conn:
            conn.execute("""
                UPDATE tasks
                SET status = ?, slurm_job_id = ?, updated_at = ?
                WHERE id = ?
            """, (status, slurm_job_id, now, task_id))
    
    def get_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific task by ID"""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT id, task_type, arguments, status, slurm_job_id, 
                   partition, created_at, updated_at, completion_file
            FROM tasks
            WHERE id = ?
        """, (task_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return self._row_to_dict(row)
    
    def get_tasks_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get all tasks with a specific status"""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT id, task_type, arguments, status, slurm_job_id,
                   partition, created_at, updated_at, completion_file
            FROM tasks
            WHERE status = ?
            ORDER BY created_at DESC
        """, (status,))
        
        return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def get_running_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks that are currently running"""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT id, task_type, arguments, status, slurm_job_id,
                   partition, created_at, updated_at, completion_file
            FROM tasks
            WHERE status LIKE 'running:%'
            ORDER BY updated_at DESC
        """)
        
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_tasks_by_slurm_job_id(self, slurm_job_id: str) -> List[Dict[str, Any]]:
        """Get all tasks matching a specific SLURM job id"""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT id, task_type, arguments, status, slurm_job_id,
                   partition, created_at, updated_at, completion_file
            FROM tasks
            WHERE slurm_job_id = ?
            ORDER BY updated_at DESC
        """, (slurm_job_id,))
        
        return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks"""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT id, task_type, arguments, status, slurm_job_id,
                   partition, created_at, updated_at, completion_file
            FROM tasks
            ORDER BY created_at DESC
        """)
        
        return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def count_tasks_by_status(self, status: str) -> int:
        """Count tasks with a specific status"""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT COUNT(*) as count
            FROM tasks
            WHERE status = ?
        """, (status,))
        
        return cursor.fetchone()['count']
    
    def count_idle_tasks(self) -> int:
        """Count idle tasks"""
        return self.count_tasks_by_status('idle')
    
    def count_running_tasks(self) -> int:
        """Count running tasks"""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT COUNT(*) as count
            FROM tasks
            WHERE status LIKE 'running:%'
        """)
        
        return cursor.fetchone()['count']
    
    def count_done_tasks(self) -> int:
        """Count completed tasks"""
        return self.count_tasks_by_status('done')
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a database row to a dictionary"""
        return {
            'id': row['id'],
            'task_type': row['task_type'],
            'arguments': json.loads(row['arguments']),
            'status': row['status'],
            'slurm_job_id': row['slurm_job_id'],
            'partition': row['partition'],
            'created_at': row['created_at'],
            'updated_at': row['updated_at'],
            'completion_file': row['completion_file']
        }
    
    def close(self):
        """Close database connection"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
