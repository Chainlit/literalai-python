#!/bin/bash

DOCS_DIR=$1

if [ -z "$DOCS_DIR" ]; then
    echo "Please provide the path to the docs directory"
    exit 1
fi

# Define your list
apiFiles=(
    "api.__init__"
)

clientFiles=(
    "client"
)

# Loop over the list
for i in "${apiFiles[@]}"; do
    echo "Generating docs for $i in api/$i.mdx"
    # make parent dir if not exists $DOCS_DIR/python-api/$i.mdx
    mkdir -p $DOCS_DIR/api

    pydoc-markdown -I . -m literalai.$i --no-render-toc > $DOCS_DIR/api/$i.mdx
done

for i in "${clientFiles[@]}"; do
    echo "Generating docs for $i in client/$i.mdx"
    # make parent dir if not exists $DOCS_DIR/python-client/$i.mdx
    mkdir -p $DOCS_DIR/client

    pydoc-markdown -I . -m literalai.$i --no-render-toc > $DOCS_DIR/client/$i.mdx
done