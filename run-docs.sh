#!/bin/bash

# ./run-docs.sh ../literal-docs/python-client/

# https://github.com/NiklasRosenstein/pydoc-markdown/blob/develop/src/pydoc_markdown/contrib/renderers/markdown.py#L52
# https://github.com/NiklasRosenstein/pydoc-markdown/blob/develop/src/pydoc_markdown/contrib/processors/filter.py#L31


DOCS_DIR=$1

if [ -z "$DOCS_DIR" ]; then
    echo "Please provide the path to the docs directory"
    exit 1
fi

# Define your list
apiFiles=(
    "api.__init__"
)
# Loop over the list
for i in "${apiFiles[@]}"; do
    echo "Generating docs for $i in api/api.mdx"
    # make parent dir if not exists $DOCS_DIR/api

    pydoc-markdown -I . -m literalai.$i --no-render-toc > $DOCS_DIR/api-reference/api.mdx
done




clientFiles=(
    "client"
)
for i in "${clientFiles[@]}"; do
    echo "Generating docs for $i in client/$i.mdx"
    # make parent dir if not exists $DOCS_DIR/python-client/$i.mdx

    pydoc-markdown -I . -m literalai.$i --no-render-toc > $DOCS_DIR/api-reference/$i.mdx
done
