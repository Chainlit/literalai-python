#!/bin/bash

# ./run-docs.sh ../literal-docs/python-client/

# https://github.com/NiklasRosenstein/pydoc-markdown/blob/develop/src/pydoc_markdown/contrib/renderers/markdown.py#L52
# https://github.com/NiklasRosenstein/pydoc-markdown/blob/develop/src/pydoc_markdown/contrib/processors/filter.py#L31


DOCS_DIR=$1
CONFIG_FILE=$2

if [ -z "$DOCS_DIR" ]; then
    echo "Please provide the path to the docs directory"
    exit 1
fi

# if no config file is provided, use the default (pydoc-markdown.yaml) (if does not exist, print config file not found)
if [ -z "$CONFIG_FILE" ]; then
    if [ -f "pydoc-markdown.yaml" ]; then
        CONFIG_FILE="pydoc-markdown.yaml"
    else
        echo "Config file not found. Please provide the path to the config file"
        exit 1
    fi
fi


# Define your list
apiFiles=(
    "api.__init__"
)

clientFiles=(
    client
    message
    step
    thread
    dataset
    dataset_item
)


mkdir -p $DOCS_DIR/api
mkdir -p $DOCS_DIR/client

for i in "${apiFiles[@]}"; do
    echo "Generating docs for $i in api/$i.mdx"

    pydoc-markdown -I . -m literalai.$i --no-render-toc > $DOCS_DIR/api/$i.mdx
done

for i in "${clientFiles[@]}"; do
    echo "Generating docs for $i in client/$i.mdx"

    pydoc-markdown -I . -m literalai.$i --no-render-toc > $DOCS_DIR/client/$i.mdx
    # python3 ../pydoc-markdown/src/pydoc_markdown/main.py -I  ~/Documents/CHAINLIT/python-client -m .$i --no-render-toc "$CONFIG_FILE" > $DOCS_DIR/client/$i.mdx
done