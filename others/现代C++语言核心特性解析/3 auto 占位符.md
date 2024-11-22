---
date: 2024-11-22 20:04:54
date modified: 2024-11-22 20:53:50
title: auto 占位符
tags:
  - cpp
categories:
  - cpp
---
## 重新定义的 auto 关键字

C++11标准赋予了auto新的含义：声明变量时根据初始化表达式自动推断该变量的类型、声明函数时函数返回值的占位符。例如：

```cpp
auto i = 5;  // 推断为 int
auto str = "hello auto"; // 推断为 const char*
auto sum(int a1, int a2) -> int { // 返回类型后置，auto为返回值占位符
	return a1 + a2;
}
```

注意，auto占位符会让编译器去推导变量类型，如果我们编写的代码让编译器无法进行推导，那么使用auto会导致编译失败。进一步来说，有4点需要引起注意。

1. 当用一个auto关键字声明多个变量的时候，编译器遵从由左往右的推导规则，以最左边的表达式推断auto的具体类型：

```cpp
int n = 5;
auto *pn = &n, m = 10;
```

但是如果写成下面的代码，将无法通过编译：

```cpp
int n = 5;
auto *pn = &n, m = 10.0; // 编译失败，声明类型不统一
```

上面两段代码唯一的区别在于赋值m的是浮点数，这和auto推导类型不匹配。

2. 当使用条件表达式初始化auto声明的变量时，编译器总是使用表达能力更强的类型：

```cpp
auto i = true ? 5 : 8.0; // i 的数据类型为 double
```

在上面的代码中，虽然能够确定表达式返回的是int类型，但是 i 的类型依旧会被推导为表达能力更强的类型double。

3. 虽然C++11标准已经支持在声明成员变量时初始化（见第8章），但是auto却无法在这种情况下声明非静态成员变量：

```cpp
struct sometype {
	auto i = 5; // 错误。无法编译通过
};
```

在C++11中静态成员变量是可以用auto声明并且初始化的，不过前提是auto必须使用const限定符：

```cpp
struct sometype {
	static const auto i = 5;
};
```

遗憾的是，const限定符会导致i常量化，显然这不是我们想要的结果。幸运的是，在C++17标准中，对于静态成员变量，auto可以在没有const的情况下使用，例如：

```cpp
struct sometype { 
	static inline auto i = 5;
};
```

按照 C++20 之前的标准，无法在函数形参列表中使用 auto 声明形参 ( 注意，在C++14中，auto 可以为 lambda 表达式声明形参)：

```cpp
void echo(auto str) { ... } // c++20 之前编译失败，c++20 编译成功
```

另外, auto 也可以和 new 关键字结合。当然我们通常不会这么用，例如

```cpp
auto i = new auto(5);
auto *j = new auto(5);
```

这种用法比较有趣，编译器实际上进行了两次推导，第一次是auto(5)，auto被推导为int类型，于是new int的类型为int\*，再通过int \*推导i和j的类型。我不建议像上面这样使用auto，因为它会破坏代码的可读性。在后面的内容中，我们将讨论应该在什么时候避免使用auto关键字。

## 推导规则

1. 如果 auto 声明的变量是按值初始化，则推导出的类型会忽略 cv 限定符。进一步解释为，在使用 auto 声明变量时，既没有使用引用，也没有使用指针，那么编译器在推导的时候会忽略 const 和 volatile 限定符。当然 auto 本身也支持添加 cv 限定符：

```cpp
const int i = 5;
auto j = i;        // auto推导类型为int，而非const int
auto &m = i;       // auto推导类型为const int，m推导类型为const int&
auto *k = i;       // auto推导类型为const int，k推导类型为const int*
const auto n = j;  // auto推导类型为int，n的类型为const int
```

2. 使用auto声明变量初始化时，目标对象如果是引用，则引用属性会被忽略：

```cpp
int i = 5;
int &j = i;
auto m = j; // auto 推导类型为 int，而非 int&
```

3. 使用auto和万能引用声明变量时（见第6章），对于左值会将auto推导为引用类型：

```cpp
int i = 5;
auto&& m = i; // auto推导类型为int& （这里涉及引用折叠的概念）
auto && j = 5; // auto推导类型为int
```

因为i是一个左值，所以m的类型被推导为int&

4. 使用auto声明变量，如果目标对象是一个数组或者函数，则auto会被推导为对应的指针类型：

```cpp
int i[5];
auto m = i; // auto 推导类型为 int*
int sum(int a1, int a2) {
	return a1 + a2;
}
auto j = sum // auto 推导类型为 int (__cdecl *)(int, int)
```

5. 当auto关键字与列表初始化组合时，这里的规则有新老两个版本，这里只介绍新规则（C++17标准）。

- 直接使用列表初始化，列表中必须为单元素，否则无法编译，auto类型被推导为单元素的类型。

- 用等号加列表初始化，列表中可以包含单个或者多个元素，auto类型被推导为`std::initializer_list<T>`，其中T是元素类型。请注意，在列表中包含多个元素的时候，元素的类型必须相同，否则编译器会报错。

```cpp
auto x1 = { 1, 2 };    // x1类型为 std::initializer_list<int>
auto x2 = { 1, 2.0 };  // 编译失败，花括号中元素类型不同
auto x3{1, 2};         // 编译失败，不是单个元素
auto x4 = { 3 };       // x4类型为std::initializer_list<int>
auto x5{ 3 };          // x5类型为int
```

可以分析一下下面的代码 auto 会被推断成什么类型

```cpp
class Base {
public:
	virtual void f() {
		std::cout << "Base::f()" << std::endl;
	};
};

class Derived : public Base {
public:
	virtual void f() override {
		std::cout << "Derived::f()" << std::endl;
	};
};
Base* d = new Derived();
auto b = *d;
b.f();
```

## 什么时候使用auto

- 当一眼就能看出声明变量的初始化类型的时候可以使用auto。

- 对于复杂的类型，例如lambda表达式、bind等直接使用auto。 

对于第一条规则，常见的是在容器的迭代器上使用，例如：

```cpp
std::map<std::string, int> str2int;
// ... 填充 str2int 的代码
for (std::map<std::string, int>::const_iterator it = str2int.cbegin(); it != str2int.cend(); ++it) {}
// 或者
for (std::pair<const std::string, int> &it : str2int) {}
```

反过来说，如果使用auto声明变量，则会导致其他程序员阅读代码时需要翻阅初始化变量的具体类型，那么我们需要慎重考虑是否适合使用auto关键字。

对于第二条规则，我们有时候会遇到无法写出类型或者过于复杂的类型，或者即使能正确写出某些复杂类型，但是其他程序员阅读起来也很费劲，这种时候建议使用auto来声明，例如lambda表达式：

```cpp
auto l = [](int a1, int a2) { return a1 + a2; };
```

再例如：

```cpp
int sum(int a1, int a2) { return a1 + a2; }
auto b = std::bind(sum, 5, std::placeholders::_1);
```

这里b的类型为`std::_Binder<std::_Unforced,int(cdecl &) (int,int),int, const std::_Ph<1> &>`，绝大多数读者看到这种类型时会默契地选择使用auto来声明变量。

## 返回类型的推导

C++14标准支持对返回类型声明为auto的推导，例如：

```cpp
auto sum(int a1, int a2) { return a1 + a2; }
```

请注意，如果有多重返回值，那么需要保证返回值类型是相同的。

```cpp
auto sum(long a1, long a2) {
	if (a1 < 0) {
		return 0; // 返回 int 类型
	} else {
		return a1 + a2; // 返回 long 类型
	}
}
```

以上代码中有两处返回，return 0返回的是int类型，而 return a1+a2 返回的是long类型，这种不同的返回类型会导致编译失败。

## lambda 表达式中使用 auto 类型推导

在C++14标准中我们还可以把auto写到lambda表达式的形参中，这样就得到了一个泛型的lambda表达式，例如：

```cpp
auto l = [](auto a1, auto a2) { return a1 + a2; };
auto retval = l(5, 5.0);
```

在上面的代码中a1被推导为int类型，a2被推导为double类型，返回值retval被推导为double类型。

```cpp
auto l = [](int &i)->auto& { return i; };
auto x1 = 5;
auto &x2 = l(x1);
assert(&x1 == &x2); // 有相同的内存地址
```

起初在后置返回类型中使用auto是不允许的，但是后来人们发现，这是唯一让lambda表达式通过推导返回引用类型的方法了。

## 非类型模版形参占位符

C++17标准对auto关键字又一次进行了扩展，使它可以作为非类型模板形参的占位符。当然，我们必须保证推导出来的类型是可以用作模板形参的，否则无法通过编译，例如：

```cpp
#include <iostream>
template<auto N>
void f() {
	std::cout << N << std::endl;
}

int main() {
	f<5>(); // N 为 int 类型
	f<'c'>(); // N 为 char 类型
	f<5.0>(); // 编译失败，模版参数不能为 double
}
```

在上面的代码中，函数f<5>()中5的类型为int，所以auto被推导为int类型。同理，f<'c'>()的auto被推导为char类型。由于f<5.0>()的5.0被推导为double类型，但是模板参数不能为double类型，因此导致编译失败。

