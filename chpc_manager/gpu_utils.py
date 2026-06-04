"""
GPU Memory Utilities
Provides functions to filter nodes based on GPU memory requirements
"""
from typing import List, Dict, Set
import os
import sys

# Import GPU specs
try:
    from gpu_specs import gpumem_node_map
except ImportError:
    # If gpu_specs.py is not in the same directory, try to import from parent
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gpu_specs import gpumem_node_map


def get_nodes_with_insufficient_memory(cluster: str, min_memory_gb: int) -> List[str]:
    """
    Get list of nodes in a cluster that have insufficient GPU memory
    
    Args:
        cluster: Cluster name (e.g., 'notchpeak', 'granite', 'lonepeak', 'kingspeak')
        min_memory_gb: Minimum required GPU memory in GB
    
    Returns:
        List of node names that should be excluded
    """
    if cluster not in gpumem_node_map:
        # Unknown cluster, return empty list (no exclusions)
        return []
    
    cluster_nodes = gpumem_node_map[cluster]
    insufficient_nodes = []
    
    for node_name, gpu_memory in cluster_nodes.items():
        if gpu_memory < min_memory_gb:
            insufficient_nodes.append(node_name)
    
    return insufficient_nodes


def get_nodes_without_gpu(cluster: str) -> List[str]:
    """
    Get list of nodes in a cluster that have no working GPU (memory = 0)
    
    Args:
        cluster: Cluster name
    
    Returns:
        List of node names with no GPU
    """
    return get_nodes_with_insufficient_memory(cluster, min_memory_gb=1)


def get_cluster_gpu_info(cluster: str) -> Dict[str, int]:
    """
    Get GPU memory info for all nodes in a cluster
    
    Args:
        cluster: Cluster name
    
    Returns:
        Dictionary mapping node names to GPU memory in GB
    """
    return gpumem_node_map.get(cluster, {})


def merge_exclusion_lists(*exclusion_lists: List[str]) -> List[str]:
    """
    Merge multiple exclusion lists, removing duplicates
    
    Args:
        *exclusion_lists: Variable number of exclusion lists
    
    Returns:
        Merged list with unique node names
    """
    # Use set to remove duplicates, then convert back to sorted list
    merged = set()
    for exclusion_list in exclusion_lists:
        if exclusion_list:
            merged.update(exclusion_list)
    
    return sorted(list(merged))


def get_task_memory_exclusions(cluster: str, task_config: dict) -> List[str]:
    """
    Get nodes to exclude based on task's GPU memory requirement
    
    Args:
        cluster: Cluster name
        task_config: Task type configuration dictionary
    
    Returns:
        List of nodes to exclude due to insufficient GPU memory
    """
    # Check if task specifies GPU memory requirement
    default_resources = task_config.get('default_resources', {})
    min_gpu_memory = default_resources.get('gpu_memory_gb')
    
    if min_gpu_memory is None:
        # No GPU memory requirement specified
        return []
    
    try:
        min_gpu_memory = int(min_gpu_memory)
    except (ValueError, TypeError):
        # Invalid memory specification
        return []
    
    # Get nodes with insufficient memory
    return get_nodes_with_insufficient_memory(cluster, min_gpu_memory)
