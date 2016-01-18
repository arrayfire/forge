// Umar Arshad
// Copyright 2015-2019
//
// Modified by Pradeep Garigipati on Dec 30, 2015 for Forge
// Purpose of modification: To use the program to convert
// GLSL shader files into compile time constant string literals

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>

using namespace std;
typedef map<string, string> opt_t;

static
void print_usage() {
    cout << R"delimiter(GLSL2CPP
Converts OpenGL shader files to C++ headers. It is similar to bin2c and xxd but adds
support for namespaces.

| --name        | name of the variable (default: var)                               |
| --file        | input file                                                        |
| --output      | output file (If no output is specified then it prints to stdout)  |
| --type        | Type of variable (default: char)                                  |
| --namespace   | A space seperated list of namespaces                              |
| --formatted   | Tabs for formatting                                               |
| --version     | Prints my name                                                    |
| --help        | Prints usage info                                                 |

Example
-------
Command:
./glsl2cpp --file blah.txt --namespace shaders --formatted --name image_vs

Will produce the following:
#pragma once
#include <cstddef>
namespace shaders {
    static const char image_vs[] = R"shader(
#version 330

layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 tex;

uniform mat4 matrix;

out vec2 texcoord;

void main() {
    texcoord = tex;
    gl_Position = matrix * vec4(pos,1.0);
}
    )shader";
}
)delimiter";
        exit(0);
}

static bool formatted;

static
void add_tabs(const int level)
{
    if(formatted) {
        for(int i =0; i < level; i++) {
            cout << "\t";
        }
    }
}

static
opt_t parse_options(const vector<string>& args)
{
    opt_t options;

    options["--name"]       = "";
    options["--type"]       = "";
    options["--file"]       = "";
    options["--output"]     = "";
    options["--namespace"]  = "";
    options["--eof"]        = "";

    //Parse Arguments
    string curr_opt;
    bool verbose = false;
    for(auto arg : args) {
        if(arg == "--verbose") {
            verbose = true;
        } else if(arg == "--formatted") {
            formatted = true;
        } else if(arg == "--version") {
            cout << args[0] << " Original Author: Umar Arshad;\n Modified later by: Pradeep Garigipati." << endl;
        } else if(arg == "--help") {
            print_usage();
        } else if(options.find(arg) != options.end()) {
            curr_opt = arg;
        } else if(curr_opt.empty()) {
            //cerr << "Invalid Argument: " << arg << endl;
        } else {
            if(options[curr_opt] != "") {
                options[curr_opt] += " " + arg;
            }
            else {
                options[curr_opt] += arg;
            }
        }
    }

    if(verbose) {
        for(auto opts : options) {
            cout << get<0>(opts) << " " << get<1>(opts) << endl;
        }
    }
    return options;
}

int main(int argc, const char * const * const argv)
{

    vector<string> args(argv, argv+argc);

    opt_t&& options = parse_options(args);

    //Save default cout buffer. Need this to prevent crash.
    auto bak = cout.rdbuf();
    unique_ptr<ofstream> outfile;

    // Set defaults
    if(options["--name"] == "")     { options["--name"]     = "var"; }
    if(options["--output"] != "")   {
        //redirect stream if output file is specified
        outfile.reset(new ofstream(options["--output"]));
        cout.rdbuf(outfile->rdbuf());
    }

    cout << "#pragma once\n";
    cout << "#include <string>\n"; // defines std::string

    int ns_cnt = 0;
    int level = 0;
    if(options["--namespace"] != "") {
        std::stringstream namespaces(options["--namespace"]);
        string name;
        namespaces >> name;
        do {
            add_tabs(level++);
            cout << "namespace " << name << "\n{\n";
            ns_cnt++;
            namespaces >> name;
        } while(!namespaces.fail());
    }

    if(options["--type"] == "") {
        options["--type"]     = "std::string";
    }
    add_tabs(level);
    cout << "static const " << options["--type"] << " " << options["--name"] << " = R\"shader(\n";
    level++;

    ifstream input(options["--file"]);

    for(std::string line; std::getline(input, line);) {
        add_tabs(level);
        cout << line << endl;
    }

    if (options["--eof"].c_str()[0] == '1') {
        // Add end of file character
        cout << "0x0";
    }

    add_tabs(--level);
    cout << ")shader\";\n";

    while(ns_cnt--) {
        add_tabs(--level);
        cout << "}\n";
    }
    cout.rdbuf(bak);
}
