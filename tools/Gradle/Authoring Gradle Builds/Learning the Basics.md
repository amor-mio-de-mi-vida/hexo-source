---
title: Learning the Basics
tags:
  - tools
  - gradle
categories:
  - Authoring Gradle Builds
date created: 2024-10-02 16:51:12
date modified: 2024-10-02 20:44:00
---
## Gradle Directories
Gradle uses two main directories to perform and manage its work: the Gradle User Home directory and the Project Root directory.
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/gradle/Pasted-image-20241002165520.syxl5e79m.webp)
### Gradle User Home directory
By default, the Gradle User Home (`~/.gradle` or `C:\Users\<USERNAME>\.gradle`) stores global configuration properties, initialization scripts, caches, and log files.



## Build Lifecycle



## Writing Settings Files
The settings file is the entry point of every Gradle build.
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/gradle/Pasted-image-20241002165822.77djgqxgd7.webp)
As the settings script executes, it configures this `Settings`. Therefore, the _settings file_ defines the `Settings` object.

Many top-level properties and blocks in a settings script are part of the Settings API.

The following table lists a few commonly used properties:

|Name|Description|
|---|---|
|`buildCache`|The build cache configuration.|
|`plugins`|The container of plugins that have been applied to the settings.|
|`rootDir`|The root directory of the build. The root directory is the project directory of the root project.|
|`rootProject`|The root project of the build.|
|`settings`|Returns this settings object.|

The following table lists a few commonly used methods:

| Name             | Description                                                    |
| ---------------- | -------------------------------------------------------------- |
| `include()`      | Adds the given projects to the build.                          |
| `includeBuild()` | Includes a build at the specified path to the composite build. |
## Write Build Scripts
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/gradle/Pasted-image-20241002170658.361k2cuymo.webp)
A **build script** configures a **project** and is associated with an object of type Project.

The following table lists a few commonly used properties:

|Name|Type|Description|
|---|---|---|
|`name`|`String`|The name of the project directory.|
|`path`|`String`|The fully qualified name of the project.|
|`description`|`String`|A description for the project.|
|`dependencies`|`DependencyHandler`|Returns the dependency handler of the project.|
|`repositories`|`RepositoryHandler`|Returns the repository handler of the project.|
|`layout`|`ProjectLayout`|Provides access to several important locations for a project.|
|`group`|`Object`|The group of this project.|
|`version`|`Object`|The version of this project.|

The following table lists a few commonly used methods:

|Name|Description|
|---|---|
|`uri()`|Resolves a file path to a URI, relative to the project directory of this project.|
|`task()`|Creates a Task with the given name and adds it to this project.|
## Using Tasks
The work that Gradle can do on a project is defined by one or more _tasks_.
![](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/gradle/Pasted-image-20241002172201.4xuix9f3tn.webp)
Gradle provides several default tasks for a project, which are listed by running
```bash
./gradlew tasks
```

Tasks either come from **build scripts** or **plugins**. Once we apply a plugin to our project, such as the `application` plugin, additional tasks become available:

To view information about a task, use 
```bash
/gradlew help --task <task-name>
```
> The calls `doFirst` and `doLast` can be executed multiple times. They add an action to the beginning or the end of the task’s actions list. When the task executes, the actions in the action list are executed in order.

## Writing Tasks
To create a task, inherit from the `DefaultTask` class and implement a [`@TaskAction`](https://docs.gradle.org/current/javadoc/org/gradle/api/tasks/TaskAction.html) handler:
```kotlin
abstract class CreateFileTask : DefaultTask() {
    @TaskAction
    fun action() {
        val file = File("myfile.txt")
        file.createNewFile()
        file.writeText("HELLO FROM MY TASK")
    }
}
```
A task is **registered** in the build script using the `TaskContainer.register()` method, which allows it to be then used in the build logic.
```kotlin
abstract class CreateFileTask : DefaultTask() {
    @TaskAction
    fun action() {
        val file = File("myfile.txt")
        file.createNewFile()
        file.writeText("HELLO FROM MY TASK")
    }
}

tasks.register<CreateFileTask>("createFileTask")
```
Setting the **group** and **description** properties on your tasks can help users understand how to use your task:
```kotlin
abstract class CreateFileTask : DefaultTask() {
    @TaskAction
    fun action() {
        val file = File("myfile.txt")
        file.createNewFile()
        file.writeText("HELLO FROM MY TASK")
    }
}

tasks.register<CreateFileTask>("createFileTask", ) {
    group = "custom"
    description = "Create myfile.txt in the current directory"
}
```
For the task to do useful work, it typically needs some **inputs**. A task typically produces **outputs**.
```kotlin
abstract class CreateFileTask : DefaultTask() {
    @Input
    val fileText = "HELLO FROM MY TASK"

    @Input
    val fileName = "myfile.txt"

    @OutputFile
    val myFile: File = File(fileName)

    @TaskAction
    fun action() {
        myFile.createNewFile()
        myFile.writeText(fileText)
    }
}

tasks.register<CreateFileTask>("createFileTask") {
    group = "custom"
    description = "Create myfile.txt in the current directory"
}
```
A task is optionally **configured** in a build script using the `TaskCollection.named()` method.
```kotlin
abstract class CreateFileTask : DefaultTask() {
    @get:Input
    abstract val fileText: Property<String>

    @Input
    val fileName = "myfile.txt"

    @OutputFile
    val myFile: File = File(fileName)

    @TaskAction
    fun action() {
        myFile.createNewFile()
        myFile.writeText(fileText.get())
    }
}

tasks.register<CreateFileTask>("createFileTask") {
    group = "custom"
    description = "Create myfile.txt in the current directory"
    fileText.convention("HELLO FROM THE CREATE FILE TASK METHOD") // Set convention
}

tasks.named<CreateFileTask>("createFileTask") {
    fileText.set("HELLO FROM THE NAMED METHOD") // Override with custom message
}
```