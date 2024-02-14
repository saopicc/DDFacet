set -e
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
#test on 22.04
docker build -t "$IMAGENAME.2004:$BUILD_NUMBER" --no-cache=true -f docker.2004 .
docker build -t "$IMAGENAME.2204:$BUILD_NUMBER" --no-cache=true -f docker.2204 .
docker run -m 100g --cap-add sys_ptrace \
		   --memory-swap=-1 \
                   --shm-size=150g \
                   --rm=true \
                   --name=$IMAGENAME$BUILD_NUMBER \
                   -v $TEST_DATA_DIR:/test_data \
                   -v $TEST_OUTPUT_DIR:/test_output \
                   -v $PROJECTS_DIR:/src \
                   -v $WORKSPACE_ROOT:/workspace \
		   --workdir /test_output \
                   --entrypoint sh \
                   $IMAGENAME.2204:$BUILD_NUMBER \
                   -c "ln -s /test_data/beams /test_output/beams && pynose -s --with-xunit --xunit-file /workspace/nosetests.xml /src/DDFacet/DDFacet/Tests"
