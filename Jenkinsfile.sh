echo "----------------------------------------------"
echo "$JOB_NAME build $BUILD_NUMBER"
WORKSPACE_ROOT="$WORKSPACE/$BUILD_NUMBER"
echo "Setting up build in $WORKSPACE_ROOT"
TEST_OUTPUT_DIR_REL=testcase_output
TEST_OUTPUT_DIR="$WORKSPACE_ROOT/$TEST_OUTPUT_DIR_REL"
TEST_DATA_DIR="$WORKSPACE/../../../test-data"
PROJECTS_DIR_REL="projects"
PROJECTS_DIR=$WORKSPACE_ROOT/$PROJECTS_DIR_REL
mkdir $TEST_OUTPUT_DIR
echo "----------------------------------------------"
echo "\nEnvironment:"
df -h .
echo "----------------------------------------------"
cat /proc/meminfo
echo "----------------------------------------------"

#build using docker file in directory:
cd $PROJECTS_DIR/DDFacet
IMAGENAME="ddf"
docker build -t "$IMAGENAME:$BUILD_NUMBER" --no-cache=true .
docker run -m 100g --cap-add sys_ptrace \
				   --memory-swap=-1 \
                   --shm-size=150g \
                   --rm=true \
                   --name=$IMAGENAME$BUILD_NUMBER \
                   -v $TEST_DATA_DIR:/test_data \
                   -v $TEST_OUTPUT_DIR:/test_output \
                   -v $PROJECTS_DIR:/src \
                   -v $WORKSPACE_ROOT:/workspace \
                   --entrypoint sh \
                   $IMAGENAME:$BUILD_NUMBER \
                   -c "nosetests -s --with-xunit --xunit-file /workspace/nosetests.xml /usr/local/lib/python2.7/dist-packages/DDFacet/Tests"