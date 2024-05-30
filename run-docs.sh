#!/bin/bash

# ./run-docs.sh ../literal-docs/python-client/

# https://github.com/NiklasRosenstein/pydoc-markdown/blob/develop/src/pydoc_markdown/contrib/renderers/markdown.py#L52
# https://github.com/NiklasRosenstein/pydoc-markdown/blob/develop/src/pydoc_markdown/contrib/processors/filter.py#L31


DOCS_DIR="generated_docs/"
CONFIG_FILE="pydoc-markdown.yaml"
BEAUTIFY=false


# same as above, but make the arguments switchable and add a --beautify flag
for i in "$@"
do
    case $i in
        -b|--beautify)
        BEAUTIFY=true
        shift # past argument=value
        ;;
        -d=*|--docs-dir=*)
        DOCS_DIR="${i#*=}"
        shift # past argument=value
        ;;
        -c=*|--config-file=*)
        CONFIG_FILE="${i#*=}"
        shift # past argument=value
        ;;
        *)
        # unknown option
        ;;
    esac
done


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
    echo "Writing docs for $i in $DOCS_DIR/$i.mdx"
    
    rm -f $DOCS_DIR/$i.*
    pydoc-markdown -I . -m literalai.$i --no-render-toc "$CONFIG_FILE" > $DOCS_DIR/$i.mdx

    if [ "$BEAUTIFY" = false ]; then
        continue
    fi

    echo "Beautifying $i.mdx"
    sed -r -E 's/- `([^`]+)` _([^_]+)_ - (.*)$/<ResponseField name="\1" type="\2">\3<\/ResponseField>/' $DOCS_DIR/$i.mdx > $DOCS_DIR/$i.tmp
    sed -r -E 's/- `([^`]+)` - (.*)$/<ResponseField name="\1">\2<\/ResponseField>/' $DOCS_DIR/$i.tmp > $DOCS_DIR/$i.mdx
    rm $DOCS_DIR/$i.tmp

    # remove the third line of the file
    sed "3d" $DOCS_DIR/$i.mdx > $DOCS_DIR/$i.tmp && mv $DOCS_DIR/$i.tmp $DOCS_DIR/$i.mdx
done

# rename the api.__init__.mdx to api.mdx
mv $DOCS_DIR/api.__init__.mdx $DOCS_DIR/api.mdx

