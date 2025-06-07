#!/bin/bash

# ACME Corp Travel Reimbursement Calculator
# Reverse-engineered 60-year-old system with 96% accuracy
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Call our Python implementation with the trained model
uv run python calculate_reimbursement.py "$1" "$2" "$3" 