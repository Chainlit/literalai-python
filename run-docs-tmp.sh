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
    echo "Generating docs for $i in api/$i.mdx"
    
    if [ "$i" == "api.__init__" ]; then
        rm -f $DOCS_DIR/api.mdx
        pydoc-markdown -I . -m literalai.$i --no-render-toc "$CONFIG_FILE" > $DOCS_DIR/api.mdx
        # use sed to replace the title from __init__ to API
        sed -i -r -E 's/# `\_\_init\_\_`/# API/' $DOCS_DIR/api.mdx
    else
        rm -f $DOCS_DIR/$i.mdx
        pydoc-markdown -I . -m literalai.$i --no-render-toc "$CONFIG_FILE" > $DOCS_DIR/$i.mdx
    fi

    if [ "$BEAUTIFY" = false ]; then
        continue
    fi

    if [ "$i" == "api.__init__" ]; then
        echo "Beautifying api.mdx"
        
        # First, transform the lines and save to a temporary file
        sed -r -E 's/- `([^`]+)` _([^_]+)_ - (.*)$/<ResponseField name="\1" type="\2">\3<\/ResponseField>/' $DOCS_DIR/api.mdx > $DOCS_DIR/api.tmp
        # Transform lines that do not have a type and save to the final file
        sed -r -E 's/- `([^`]+)` - (.*)$/<ResponseField name="\1">\2<\/ResponseField>/' $DOCS_DIR/api.tmp > $DOCS_DIR/api.tmp2
        # Remove all double quotes in the name and type sections (if any)
        tr -d '"' < $DOCS_DIR/api.tmp2 > $DOCS_DIR/api.mdx

        # Clean up temporary files
        rm $DOCS_DIR/api.tmp $DOCS_DIR/api.tmp2

        continue
    fi

    echo "Beautifying $i.mdx"
    sed -r -E 's/- `([^`]+)` _([^_]+)_ - (.*)$/<ResponseField name="\1" type="\2">\3<\/ResponseField>/' $DOCS_DIR/$i.mdx > $DOCS_DIR/$i.tmp
    sed -r -E 's/- `([^`]+)` - (.*)$/<ResponseField name="\1">\2<\/ResponseField>/' $DOCS_DIR/$i.tmp > $DOCS_DIR/$i.mdx

done

