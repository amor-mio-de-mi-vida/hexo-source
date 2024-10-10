---
date created: 2024-10-05 17:15:52
date modified: 2024-10-06 11:10:26
title: Make
tags:
  - make
  - tools
categories:
---
## Example
```make
objects = main.o kbd.o command.o display.o \
          insert.o search.o files.o utils.o

edit : $(objects)
        cc -o edit $(objects)

$(objects) : defs.h
kbd.o command.o files.o : command.h
display.o insert.o search.o files.o : buffer.h

.PHONY : clean
clean :
        -rm edit $(objects)
```
注：最后一个命令前面的`-`指忽略报错信息执行该命令
## Writing Makefile
### Makefile中的内容
- `explicit rule`说明了何时以及如何重新制作一个或多个文件，这些文件被称为 `target `。它列出了 `target` 所依赖的其他文件，这些文件被称为 `prerequisites` ，并且可能还提供了用于创建或更新` target` 的命令。
- `implicit rule` 说明了何时以及如何根据文件名重新制作一类文件。它描述了 `target `如何可能依赖于一个与 `target` 名称相似的文件，并给出了创建或更新此类 `target` 的配方。
- `variable definition`是一个指定文本字符串值的行，该值可以稍后替换到文本中。
- `directive`是一条指示make在读取Makefile时执行特殊操作的指令。这些包括：
	- 读取另一个 makefile 文件
	- 决定（基于变量值）是否使用或忽略Makefile的一部分
	- 从包含多行的字符串定义一个变量
### 引用其他makefile文件
include 指令告诉 make 暂停读取当前的 makefile，并在继续之前读取一个或多个其他 makefile。该指令是 makefile 中的一行，如下所示：
```make
include filenames...
```
使用 include 指令的一个场合是，由各个目录中的各个 makefile 处理的多个程序需要使用一组通用的变量定义或模式规则。
另一个这样的场合是当您想要从源文件自动生成先决条件时；先决条件可以放在主 makefile 包含的文件中。这种做法通常比以某种方式将先决条件附加到主 makefile 末尾的做法更干净，就像其他版本的 make 传统上所做的那样。
如果指定的名称不是以斜杠开头，并且在当前目录中找不到该文件，则会搜索其他几个目录。首先，搜索您使用“-I”或“--include-dir”选项指定的任何目录。然后按以下顺序搜索以下目录（如果存在）：prefix/include（通常为 /usr/local/include 1）/usr/gnu/include、/usr/local/include、/usr/include。 INCLUDE_DIRS 变量将包含 make 将搜索包含文件的当前目录列表。