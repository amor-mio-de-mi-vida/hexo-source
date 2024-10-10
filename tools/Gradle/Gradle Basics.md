---
title: Gradle Basics
tags:
  - tools
  - gradle
categories: 
date created: 2024-10-02 15:32:31
date modified: 2024-10-10 22:46:17
date: 2024-10-02 15:32:31
---
Gradle **automates building, testing, and deployment of software** from information in **build scripts**.
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/gradle/basic0.5c0yo4hccn.webp)
## Gradle Project structure
A Gradle project will look similar to the following:
```text
project
├── gradle                             (1)        
│   ├── libs.versions.toml             (2)
│   └── wrapper
│       ├── gradle-wrapper.jar
│       └── gradle-wrapper.properties
├── gradlew                            (3)
├── gradlew.bat                        (3) 
├── settings.gradle(.kts)              (4) 
├── subproject-a                   
│   ├── build.gradle(.kts)             (5)     
│   └── src                            (6) 
└── subproject-b                        
    ├── build.gradle(.kts)             (5) 
    └── src                            (6) 
```
(1)  Gradle directory to store wrapper files and more
(2) Gradle version catalog for dependency management
(3) Gradle wrapper scripts
(4) Gradle settings file to define a root project name and subprojects
(5) Gradle build scripts of the two subprojects - `subproject-a` and `subproject-b`
(6) Source code and/or additional files for the projects

Gradle Wrapper
The Wrapper is a script that invokes a declared version of Gradle and is **the recommended way to execute a Gradle build**. It is found in the project root directory as a `gradlew` or `gradlew.bat` file:
```bash
gradlew build     // Linux or OSX
gradlew.bat build  // Windows
```
## Gradle Wrapper Basics
The **recommended way to execute any Gradle build** is with the Gradle Wrapper.
```text
.
├── gradle 
│   └── wrapper
│       ├── gradle-wrapper.jar         (1)
│       └── gradle-wrapper.properties  (2)
├── gradlew  (3)
└── gradlew.bat (4)
```
(1) `gradle-wrapper.jar`: This is a small JAR file that contains the Gradle Wrapper code. It is responsible for downloading and installing the correct version of Gradle for a project if it’s not already installed.
(2) `gradle-wrapper.properties`: This file contains configuration properties for the Gradle Wrapper, such as the distribution URL (where to download Gradle from) and the distribution type (ZIP or TARBALL).
(3) `gradlew`: This is a shell script (Unix-based systems) that acts as a wrapper around `gradle-wrapper.jar`. It is used to execute Gradle tasks on Unix-based systems without needing to manually install Gradle.
(4) `gradlew.bat`: This is a batch script (Windows) that serves the same purpose as `gradlew` but is used on Windows systems.

If you want to view or update the Gradle version of your project, use the command line. Do not edit the wrapper files manually:
```text
./gradlew --version
./gradlew wrapper --gradle-version 7.2
```
## Command-Line Interface Basics
Substitute `./gradlew` (in macOS / Linux) or `gradlew.bat` (in Windows) for `gradle` in the following examples.

Executing Gradle on the command line conforms to the following structure:

```bash
gradle [taskName...] [--option-name...]
```
Options are allowed _before_ and _after_ task names.

```bash
gradle [--option-name...] [taskName...]
```
If multiple tasks are specified, you should separate them with a space.

```bash
gradle [taskName1 taskName2...] [--option-name...]
```
Options that accept values can be specified with or without `=` between the option and argument. The use of `=` is recommended.

```bash
gradle [...] --console=plain
```
Options that enable behavior have long-form options with inverses specified with `--no-`. The following are opposites.

```bash
gradle [...] --build-cache
gradle [...] --no-build-cache
```
Many long-form options have short-option equivalents. The following are equivalent:
```bash
gradle --help
gradle -h
```
To execute a task called `taskName` on the root project, type:
```bash
gradle :taskName
```
To pass an option to a task, prefix the option name with `--` after the task name:

```bash
gradle taskName --exampleOption=exampleValue
```
## Settings File Basics

```gradle
rootProject.name = 'root-project'   (1)

include('sub-project-a')            (2)
include('sub-project-b')
include('sub-project-c')
```
(1) Define the project name
(2) Add subprojects
## Build file Basics
```groovy
plugins {
    id 'application'                
}

application {
    mainClass = 'com.example.Main'  
}
```
(1) Add plugins
	Plugins extend Gradle’s functionality and can contribute tasks to a project.
(2) Use convention properties
	A plugin adds tasks to a project. It also adds properties and methods to a project.The `application` plugin defines tasks that package and distribute an application, such as the `run` task.
## Dependency Management Basics
To add a dependency to your project, specify a dependency in the dependencies block of your `build.gradle(.kts)` file.
Dependencies in Gradle are grouped by **configurations**.
You can view your dependency tree in the terminal using the following command
```bash
./gradlew :app:dependencies
```
## Task Basics
A task represents some **independent unit of work** that a build performs, such as compiling classes, creating a JAR, generating Javadoc, or publishing archives to a repository.
You run a Gradle `build` task using the `gradle` command or by invoking the Gradle Wrapper (`./gradlew` or `gradlew.bat`) in your project directory:
```bash
./gradlew build
```
You can list all the available tasks in the project by running the following command in the terminal:
```bash
./gradlew tasks
```
Many times, a task requires another task to run first. Build scripts can optionally define task dependencies. Gradle then automatically determines the task execution order.
## Plugin Basics
Gradle is built on a plugin system. Gradle itself is primarily composed of infrastructure, such as a sophisticated dependency resolution engine. The rest of its functionality comes from plugins. A plugin is a piece of software that **provides additional functionality to the Gradle build system**.

Plugins can be applied to a Gradle build script to **add new tasks, configurations, or other build-related capabilities**:
- Core plugins
	Gradle Core plugins are a set of plugins that are included in the Gradle distribution itself. These plugins provide essential functionality for building and managing projects.
	There are many [Gradle Core Plugins](https://docs.gradle.org/current/userguide/plugin_reference.html#plugin_reference) users can take advantage of.
- Community plugins
	Community plugins are plugins developed by the Gradle community, rather than being part of the core Gradle distribution. These plugins provide additional functionality that may be specific to certain use cases or technologies.
	Community plugins can be published at the [Gradle Plugin Portal](http://plugins.gradle.org/), where other Gradle users can easily discover and use them.
- Local plugins
	Custom or local plugins are developed and used within a specific project or organization. These plugins are not shared publicly and are tailored to the specific needs of the project or organization.

## Incremental Builds and Build Caching
Incremental builds are always enabled, and the best way to see them in action is to turn on _verbose mode_. With verbose mode, each task state is labeled during a build:
```bash 
./gradlew compileJava --console=verbose
```
The build cache stores previous build results and restores them when needed. It prevents the redundant work and cost of executing time-consuming and expensive processes.
```bash
./gradlew compileJava --build-cache
```
## Build Scan
To enable build scans on a gradle command, add `--scan` to the command line option:
```bash
 ./gradlew build --scan
```
## Reference
[Gradle Wrapper reference](https://docs.gradle.org/current/userguide/gradle_wrapper.html#gradle_wrapper_reference)
[Gradle Command Line Interface reference](https://docs.gradle.org/current/userguide/command_line_interface.html#command_line_interface)
[Writing Settings File](https://docs.gradle.org/current/userguide/writing_settings_files.html#writing_settings_files)
[Writing Build Scripts](https://docs.gradle.org/current/userguide/writing_build_scripts.html#writing_build_scripts)
[Dependency Management chapter](https://docs.gradle.org/current/userguide/glossary.html#dependency_management_terminology)
[Plugin development chapter](https://docs.gradle.org/current/userguide/custom_plugins.html#custom_plugins)
[Build cache chapter](https://docs.gradle.org/current/userguide/build_cache.html#build_cache)