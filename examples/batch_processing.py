#!/usr/bin/env python
"""
Batch processing example for SKAI-NotiAssistance.

This script demonstrates how to use the SKAI-NotiAssistance agent for processing
multiple equipment notifications in batch mode.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent import NotiAssistanceAgent
from utils.logger import setup_logging
from lib.utils import save_json

# Set up logging
setup_logging()


def load_notifications(file_path: str) -> List[Dict[str, Any]]:
    """
    Load notifications from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing notifications
        
    Returns:
        List of notification dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            notifications = json.load(f)
        
        print(f"Loaded {len(notifications)} notifications from {file_path}")
        return notifications
    
    except Exception as e:
        print(f"Error loading notifications: {str(e)}")
        return []


def process_notifications(
    agent: NotiAssistanceAgent,
    notifications: List[Dict[str, Any]],
    output_dir: str
) -> Dict[str, Any]:
    """
    Process a list of notifications and save results.
    
    Args:
        agent: NotiAssistanceAgent instance
        notifications: List of notification dictionaries
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with processing statistics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize statistics
    stats = {
        "total": len(notifications),
        "successful": 0,
        "failed": 0,
        "total_time": 0,
        "average_time": 0,
        "start_time": time.time()
    }
    
    # Process each notification
    for i, notification in enumerate(notifications):
        print(f"\nProcessing notification {i+1}/{len(notifications)}")
        print(f"Equipment ID: {notification.get('equipment_id', 'Unknown')}")
        print(f"Message: {notification.get('message', 'No message')}")
        
        # Add task type if not specified
        if "task" not in notification:
            notification["task"] = "notification_analysis"
        
        # Process the notification
        start_time = time.time()
        result = agent.run(notification)
        end_time = time.time()
        
        # Update statistics
        processing_time = end_time - start_time
        stats["total_time"] += processing_time
        
        if "error" in result and result["error"]:
            print(f"Error: {result['error']}")
            stats["failed"] += 1
        else:
            print(f"Successfully processed in {processing_time:.2f} seconds")
            stats["successful"] += 1
        
        # Save the result
        output_file = os.path.join(
            output_dir, 
            f"{notification.get('equipment_id', f'notification_{i}')}.json"
        )
        
        save_json(
            data={
                "notification": notification,
                "result": result,
                "processing_time": processing_time
            },
            file_path=output_file
        )
        
        print(f"Saved result to {output_file}")
    
    # Finalize statistics
    stats["end_time"] = time.time()
    stats["total_elapsed_time"] = stats["end_time"] - stats["start_time"]
    
    if stats["successful"] > 0:
        stats["average_time"] = stats["total_time"] / stats["successful"]
    
    # Save statistics
    stats_file = os.path.join(output_dir, "processing_stats.json")
    save_json(stats, stats_file)
    print(f"\nSaved processing statistics to {stats_file}")
    
    return stats


def main():
    """Run the batch processing example."""
    parser = argparse.ArgumentParser(description="Batch process equipment notifications")
    parser.add_argument(
        "--input", "-i", 
        default="notifications.json",
        help="Input JSON file containing notifications"
    )
    parser.add_argument(
        "--output-dir", "-o", 
        default="./output",
        help="Output directory for results"
    )
    args = parser.parse_args()
    
    print("SKAI-NotiAssistance Batch Processing Example")
    print("-------------------------------------------")
    
    # Load notifications
    notifications = load_notifications(args.input)
    
    if not notifications:
        print("No notifications to process. Exiting.")
        return
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"batch_run_{timestamp}")
    
    # Initialize the agent
    agent = NotiAssistanceAgent()
    
    # Process notifications
    print(f"\nProcessing {len(notifications)} notifications...")
    stats = process_notifications(agent, notifications, output_dir)
    
    # Print summary
    print("\nProcessing Summary")
    print("-----------------")
    print(f"Total notifications: {stats['total']}")
    print(f"Successfully processed: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total processing time: {stats['total_time']:.2f} seconds")
    print(f"Average processing time: {stats['average_time']:.2f} seconds per notification")
    print(f"Total elapsed time: {stats['total_elapsed_time']:.2f} seconds")
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main() 