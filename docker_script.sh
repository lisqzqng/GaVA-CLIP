docker run -it --gpus all \
    --name test1 \
    --runtime=nvidia \
    --shm-size=8G \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -e GDK_SCALE \
    -e GDK_DPI_SCALE \
    -v /home/dw/Documents/Code/VT-CLIP:/vlm \
    -w /vlm \
    gavaclip