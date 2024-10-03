docker run --rm -it --net host --ulimit memlock=-1 --ulimit stack=67108864 \
    --security-opt=label=disable --security-opt seccomp=unconfined \
    --tmpfs /tmp:exec --user root \
    --gpus "device=0" \
    --ipc=host \
    -p8000:8000 -p8001:8001 -p8002:8002 \
    --name lw_latest \
    -v /home/chris/engines:/engines \
    -v /home/chris/latest_trt_backend/tensorrtllm_backend:/trtback \
    -v /home/chris/latest_trt_backend/reqs/:/pipMeUp \
    llama_whisper tritonserver --model-repository=/trtback/all_models/inflight8x7/ \
    --model-control-mode=explicit \
    --load-model=whisper \
    --load-model=ensemble \
    --load-model=preprocessing \
    --load-model=postprocessing \
    --load-model=tensorrt_llm \
    --log-verbose=2 \
    --log-info=1 \
    --log-warning=1 \
    --log-error=1 \
    --http-port=8000 \
    --grpc-port=8001 \
    --metrics-port=8002
