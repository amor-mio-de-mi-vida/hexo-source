---
date: 2024-11-20 18:13:38
date modified: 2024-11-20 21:58:58
title: Welcome to use Hexo Theme Keep
tags:
  - Hexo
  - Keep
categories:
  - Hexo
---
# 值类别

## 不必要的复制

我们考虑将字符串塞入 vector 这一过程：

```cpp
int main() {
	std::vector<std::string> vec;
	vec.reserve(3);
	for (int i = 0; i < 3; i++) {
		std::string str;
		std::cin >> str;
		vec.push_back(str);
	}
	return 0;
}
```

可以发现字符串在转移的过程中，在 `str` 和 `vec` 中各保存了一份，内存占用加倍。

如果非要省下这一部分的内存，我们可以实现一个简陋的移动操作：自定义 `MyString` 结构体，内有一指针指向我们的字符串，即我们只需要把指针复制过去，并小心地清理原对象的指针，防止被错误析构。

```cpp
struct MyString {
	char *beg, *end;
	// ...
};

void move_to(MyString &src, Mystring &dst) {
	dst.beg = src.beg;
	dst.end = src.end;
	src.beg = src.end = nullptr;
}
```

由于这种高效转移对象的需求较为常见，且与 C++ 的构造、析构等操作交互困难，C++11 将移动语义引入了语言核心。

## C语言中的值类别

在 C 语言标准中，对象是一个比变量更为一般化的概念，它指代一块内存区域，具有内存地址。对象的主要属性包括：大小、有效类型、值和标识符。标识符即变量名，值是该内存以其类型解释时的含义。例如，`int` 和 `float` 类型虽然都占用 4 字节，但对于同一块内存，我们会解释出不同的含义。

C 语言中每个表达式都具有类型和值类别。值类别主要分为三类：

- 左值（lvalue）：隐含指代一个对象的表达式。即我们可以对该表达式取地址。

- 右值（rvalue）：不指代对象的表达式，即指代没有存储位置的值，我们无法取该值的地址。

- 函数指代符：函数类型的表达式。

因此，只有可修改的左值（没有 `const` 修饰且非数组的左值）可以位于赋值表达式左侧。

对于某个要求右值作为它的操作数的运算符，每当左值被用作操作数，都会对该表达式应用左值到右值，数组到指针，或者函数到指针标准转换以将它转换成右值。

常见误区：

- 右值表达式继续运算可能是左值。例如 `int *a`，表达式 `a + 1` 是右值，但 `*(a + 1)` 是左值。

- 表达式才有值类别，变量没有。例如 `int *a`，不能说变量 `a` 是左值，可以说其在表达式 `a` 中做左值，

## C++98中的值的类别

C++98 在值类别方面与 C 语言几乎一致，但增加了一些新的规则：

- 函数为左值，因为可以取地址。

- 左值引用（T&）是左值，因为可以取地址。

- 仅有 `const T&` 可绑定到右值。

## 复制消除

C++ 允许编译器执行复制消除（Copy Elision），可以减少临时对象的创建和销毁。

例如下面的代码，就触发了复制消除中的返回值优化（Return Value Optimization，RVO），你只会看到一次构造和一次复制构造，即便构造与析构有副作用。

```cpp
struct X {
	X() { std::puts("X::X()"); }
	X(const X &) { std::puts("X::X(const X &)"); }
	~X() { std::puts("X::~X()"); }
};


X get() {
	X x;
	return x;
}

int main() {
	X x = get();
	X y = X(X(X(X(x))));
	return 0;
}
```

# Lambda 表达式

下面是 Lambda 表达式的语法：

```c++
[capture](parameters) mutable -> return-type {statement}
```

下面我们分别对其中的 `capture`, `parameters`, `mutable`, `return-type`, `statement` 进行介绍。

## `capture`捕获字句

Lambda 表达式以 capture 子句开头，它指定哪些变量被捕获，以及捕获是通过值还是引用：有 `&` 符号前缀的变量通过引用访问，没有该前缀的变量通过值访问。空的 capture 子句 `[]` 指示 Lambda 表达式的主体不访问封闭范围中的变量。

我们也可以使用默认捕获模式：`&` 表示捕获到的所有变量都通过引用访问，`=` 表示捕获到的所有变量都通过值访问。之后我们可以为特定的变量 **显式** 指定相反的模式。

默认捕获时，会捕获 Lambda 中提及的变量。获的变量成为 Lambda 的一部分；与函数参数相比，调用 Lambda 时不必传递它们。

以下是一些常见的例子

```c++
int a = 0;
auto f = []() { return a * 9; }; // Error, 无法访问 'a'
auto f = [a]() { return a * 9; }; // OK, 'a' 被值「捕获」
auto f = [&a]() { return a++; }; // OK, 'a' 被引用「捕获」
								 // 注意：请保证 Lambda 被调用时 a 没有被销毁
auto b = f();  // f 从捕获列表里获得a的值，因此无需通过参数传入a
```

## `parameters` 参数列表

大多数情况下类似于函数的参数列表，例如：

```c++
auto lam = [](int a, int b) { return a + b; }
std::cout << lam(1, 9) << " " << lam(2, 6) << std::endl;
```

c++14 中，若参数类型是泛型，则可以使用auto声明类型：

```cpp
auto lam = [](auto a, auto b)
```

一个例子

```cpp
int x[] = {5, 1, 7, 6, 1, 4, 2};
std::sort(x, x + 7, [](int a, int b) { return (a > b); });
for (auto i: x) std::cout << i << " ";
```

这将打印出 `x` 数组从大到小排序后的结果。

由于 **parameters 参数列表** 是可选的，如果不将参数传递给 Lambda 表达式，并且其 Lambda 声明器不包含 mutable，且没有后置返回值类型，则可以省略空括号。

Lambda 表达式也可以将另一个 Lambda 表达式作为其参数。

```cpp
#include <functional>
#include <iostream>

int main() {
	using namespace std;
	// 返回另一个计算两数之和 Lambda 表达式
	auto addtwointegers = [](int x) -> function<int(int)> {
		return [=](int y) { return x + y; };
	};
	
	// 接受另外一个函数 f 作为参数，返回 f(z) 的两倍
	auto higherorder = [](const function<int(int)>& f, int x) {
		return f(z) * 2;
	};
	
	// 调用绑定到 higherorder 的 Lambda 表达式
	auto answer = higheroder(addtwointegers(7), 8);
	
	// 答案为 (7 + 8) * 2 = 30
	cout << answer << endl;
}
```

## `mutable` 可变规范

利用可变规范，Lambda 表达式的主体可以修改通过值捕获的变量。若使用此关键字，则 parameters **不可省略**（即使为空）。

一个例子，使用 **capture 捕获字句** 中的例子，来观察 a 的值的变化：

```cpp
int a = 0;
auto func = [a]() mutable { ++a; };
```

此时 lambda 中的 a 的值改变为 1，lambda 外的 a 保持不变。

`return-type` 返回类型

用于指定 Lambda 表达式的返回类型。若没有指定返回类型，则返回类型将被自动推断（行为与用 `auto` 声明返回值的普通函数一致）。具体的，如果函数体中没有 `return` 语句，返回类型将被推导为 `void`，否则根据返回值推导。若有多个 `return` 语句且返回值类型不同，将产生编译错误。

例如，上文的 `lam` 也可以写作：

```cpp
auto lam = [](int a, int b) -> int
```

再举两个例子：

```cpp
auto x1 = [](int i) { return i; }; // OK
auto x2 = [] { return {1, 2}; }; // Error, 返回类型被推导为 void
```

## `statement Lambda` 主体

Lambda 主体可包含任何函数可包含的部分。普通函数和 Lambda 表达式主体均可访问以下变量类型：

- 从封闭范围捕获变量

- 参数

- 本地声明的变量 

- 在一个 `class` 中声明时，若捕获 `this`, 则可以访问该对象的成员

- 具有静态存储时间的任何变量，如全局变量

下面是一个例子

```cpp
#include <iostream>

int main() {
	int m = 0, n = 0;
	[&, n](int a) mutable { m = (++n) + a; }(4);
	std::cout << m << " " << n << std::endl;
	return 0;
}
```

最后我们得到输出 `5 0`。这是由于 `n` 是通过值捕获的，在调用 Lambda 表达式后仍保持原来的值 `0` 不变。`mutable` 规范允许 `n` 在 Lambda 主体中被修改，将 `mutable` 删去则编译不通过。