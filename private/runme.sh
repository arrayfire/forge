#! /bin/bash
cd "$(dirname $0)"
cd ..
git reset --hard HEAD~10
git push -f origin master

cd examples/opencl
echo "PWD=$PWD"
if ! grep -q "cl::Context" "plot3.cpp"; then
    cmd="sed -i 's/releaseGLBuffer(handle);/releaseGLBuffer(handle);\n\tcontext = cl::Context();/g' *.cpp"
    echo "$cmd"
    eval "$cmd"
else
    echo "no need to do sed"
fi
git add * -v
git commit -m "add context = cl::Context() to avoid the program to quit with segmentation fault"
git push

