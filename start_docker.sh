docker run \
    -it \
    --ipc=host \
    -p=9999:9999 \
    --name="monai_030_rc4" \
    -v ./:/ievnet/ projectmonai/monai:0.3.0rc4 /bin/bash