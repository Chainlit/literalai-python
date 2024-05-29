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


inputFiles=(
    "api.__init__"
    "client"
    "message"
    "step"
    "thread"
    "dataset"
    "dataset_item"
)


mkdir -p $DOCS_DIR

# read all the files in the api directory and generate the docs
for i in "${inputFiles[@]}"; do
    echo "Generating docs for $i in api/$i.mdx"
    
    if [ "$i" == "api.__init__" ]; then
        rm -f $DOCS_DIR/api.mdx
        pydoc-markdown -I . -m literalai.$i --no-render-toc "$CONFIG_FILE" > $DOCS_DIR/api.mdx
    else
        rm -f $DOCS_DIR/$i.mdx
        pydoc-markdown -I . -m literalai.$i --no-render-toc "$CONFIG_FILE" > $DOCS_DIR/$i.mdx
    fi
done
