#!/usr/bin/env python
"""
Basic inference example for SKAI-NotiAssistance.

This script demonstrates how to use the SKAI-NotiAssistance agent for analyzing
equipment notifications.
"""

import os
import sys
import json
import time

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent import NotiAssistanceAgent
from utils.logger import setup_logging

# Set up logging
setup_logging()


def main():
    """Run the basic inference example."""
    print("SKAI-NotiAssistance Basic Inference Example")
    print("-------------------------------------------")
    
    # Initialize the agent
    agent = NotiAssistanceAgent()
    
    # Example notification for analysis
    notification = {
        "task": "notification_analysis",
        "equipment_id": "PUMP-101",
        "notification_type": "Warning",
        "message": "Unusual vibration detected in bearing assembly",
        "timestamp": "2023-05-15T14:30:00",
        "additional_data": "Vibration frequency: 120Hz, Amplitude: 15mm/s"
    }
    
    print(f"\nAnalyzing notification for {notification['equipment_id']}:")
    print(f"Message: {notification['message']}")
    print(f"Type: {notification['notification_type']}")
    print(f"Additional Data: {notification['additional_data']}")
    print("\nProcessing...\n")
    
    # Start time
    start_time = time.time()
    
    # Run the agent
    result = agent.run(notification)
    
    # End time
    end_time = time.time()
    
    # Print the analysis
    if "error" in result and result["error"]:
        print(f"Error: {result['error']}")
    else:
        print("Analysis:")
        print("---------")
        print(result["analysis"])
        
        print("\nStructured Data:")
        print("---------------")
        print(json.dumps(result["structured_data"], indent=2))
    
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")
    
    # Example direct query
    print("\n\nDirect Query Example")
    print("-------------------")
    
    query = {
        "task": "direct_query",
        "query": "What are common causes of unusual vibration in pumps?"
    }
    
    print(f"Query: {query['query']}")
    print("\nProcessing...\n")
    
    # Run the direct query
    query_result = agent.run(query)
    
    if "error" in query_result and query_result["error"]:
        print(f"Error: {query_result['error']}")
    else:
        print("Response:")
        print("---------")
        print(query_result["response"])
    
    print("\nDone!")


if __name__ == "__main__":
    main() 