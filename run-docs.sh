#!/bin/bash

# ./run-docs.sh ../literal-docs/python-client/

# https://github.com/NiklasRosenstein/pydoc-markdown/blob/develop/src/pydoc_markdown/contrib/renderers/markdown.py#L52
# https://github.com/NiklasRosenstein/pydoc-markdown/blob/develop/src/pydoc_markdown/contrib/processors/filter.py#L31


DOCS_DIR="../literal-docs/python-client"
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


apiFiles=(
    "api.__init__"
    "client"
)

clientFiles=(
    "message"
    "step"
    "thread"
    "dataset"
    "dataset_item"
)

mkdir -p $DOCS_DIR

# read all the files in the api directory and generate the docs
for i in "${apiFiles[@]}"; do
    echo "Writing docs for $i in $DOCS_DIR/api-reference/$i.mdx"
    
    rm -f $DOCS_DIR/api-reference/$i.*
    pydoc-markdown -I . -m literalai.$i --no-render-toc "$CONFIG_FILE" > $DOCS_DIR/api-reference/$i.mdx

    if [ "$BEAUTIFY" = false ]; then
        continue
    fi

    echo "Beautifying $i.mdx"
    sed -r -E 's/- `([^`]+)` _([^_]+)_ - (.*)$/<ResponseField name="\1" type="\2">\3<\/ResponseField>/' $DOCS_DIR/api-reference/$i.mdx > $DOCS_DIR/api-reference/$i.tmp
    sed -r -E 's/- `([^`]+)` - (.*)$/<ResponseField name="\1">\2<\/ResponseField>/' $DOCS_DIR/api-reference/$i.tmp > $DOCS_DIR/api-reference/$i.mdx
    rm $DOCS_DIR/api-reference/$i.tmp

    # remove the third line of the file
    sed "3d" $DOCS_DIR/api-reference/$i.mdx > $DOCS_DIR/api-reference/$i.tmp && mv $DOCS_DIR/api-reference/$i.tmp $DOCS_DIR/api-reference/$i.mdx

    # if the file is api.__init__, rename it to api.mdx
    if [ "$i" = "api.__init__" ]; then
        mv $DOCS_DIR/api-reference/$i.mdx $DOCS_DIR/api-reference/api.mdx
    fi
done

# same for the client files in the "abstractions" directory
for i in "${clientFiles[@]}"; do
    echo "Writing docs for $i in $DOCS_DIR/abstractions/$i.mdx"
    
    rm -f $DOCS_DIR/abstractions/$i.*
    pydoc-markdown -I . -m literalai.$i --no-render-toc "$CONFIG_FILE" > $DOCS_DIR/abstractions/$i.mdx

    if [ "$BEAUTIFY" = false ]; then
        continue
    fi

    echo "Beautifying $i.mdx"
    sed -r -E 's/- `([^`]+)` _([^_]+)_ - (.*)$/<ResponseField name="\1" type="\2">\3<\/ResponseField>/' $DOCS_DIR/abstractions/$i.mdx > $DOCS_DIR/abstractions/$i.tmp
    sed -r -E 's/- `([^`]+)` - (.*)$/<ResponseField name="\1">\2<\/ResponseField>/' $DOCS_DIR/abstractions/$i.tmp > $DOCS_DIR/abstractions/$i.mdx
    rm $DOCS_DIR/abstractions/$i.tmp

    # remove the third line of the file
    sed "3d" $DOCS_DIR/abstractions/$i.mdx > $DOCS_DIR/abstractions/$i.tmp && mv $DOCS_DIR/abstractions/$i.tmp $DOCS_DIR/abstractions/$i.mdx
done
