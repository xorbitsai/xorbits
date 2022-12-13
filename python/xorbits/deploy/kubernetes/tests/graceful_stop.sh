#!/bin/bash
set -e
touch /tmp/stopping.tmp
if [[ -f /tmp/xorbits-service.pid ]]; then
  SERVICE_PID="$(cat /tmp/xorbits-service.pid)"
  kill -INT "$SERVICE_PID" || true
  CNT=0
  while kill -0 "$SERVICE_PID"; do
    sleep 0.5
    CNT=$((CNT+1))
    if [[ $CNT -gt 10 ]]; then
      break
    fi
  done
  kill -INT "$SERVICE_PID" || true
fi
