#!/bin/sh
curl -v -X POST http://127.0.0.1:8080/invocations \
   -H 'Content-Type: application/json' \
   -d '{ "prompt": "marilyn monroe singing for president kennedy", "num_inference_steps": 75, "guidance_scale": 7.5 }' \
   -o maui.jpg