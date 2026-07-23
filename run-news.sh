#!/bin/bash

# input parameter check
if [ -z "$1" ]; then
  echo "[$(date)] ERROR: No argument provided. Expected: Morning or Evening." >> /home/naka-h/.local/logs/news.log
  exit 1
fi

LOG="$HOME/.local/logs/news.log"
EVAL="./eval-data/eval-prompt.csv"

cd $HOME/git/personalized-news-agent/

echo "=== $(date) $1 run ===" >> $LOG

.venv/bin/python ./news-agent.py "$1" --eval >> $LOG 2>&1
.venv/bin/python ./eval-prompt.py -i latest -o $EVAL -m all >> $LOG 2>&1

