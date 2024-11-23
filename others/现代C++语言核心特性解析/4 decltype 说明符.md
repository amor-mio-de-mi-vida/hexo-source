---
date: 2024-11-22 20:54:18
date modified: 2024-11-23 10:10:55
title: decltype 说明符
tags:
  - cpp
categories:
  - cpp
---
## 回顾 typeof 和 typeid

GCC的扩展提供了一个名为typeof的运 算符。通过该运算符可以获取操作数的具体类型。这让使用GCC的程序 员在很早之前就具有了对对象类型进行推导的能力。

```cpp
int a = 0;
typeof(a) b = 5;
```

除使用GCC提供的typeof运算符获取对象类型以外，C++标准还 提供了一个typeid运算符来获取与目标操作数类型有关的信息。获 取的类型信息会包含在一个类型为std::type_info的对象里。我们 可以调用成员函数name获取其类型名，例如:

```cpp
int x1 = 0;
double x2 = 5.5;
std::cout << typeid(x1).name() << std::endl;
std::cout << typeid(x1 + x2).name() << std::endl;
std::cout << typeid(int).name() << std::endl;
```

有三点需要注意：

- typeid的返回值是一个左值，且其生命周期一直被扩展到程序生命周期结束。

- typeid返回的std::type_info删除了复制构造函数，若想 保存std::type_info，只能获取其引用或者指针，例如:

```cpp
auto t1 = typeid(int);  // 编译失败，没有复制构造函数无法编译
auto &t2 = typeid(int); // 编译成功，t2推导为 const std::type_info&
auto t3 = &typeid(int); // 编译成功，t3推导为 const std::type_info*
```

- typeid的返回值总是忽略类型的 cv 限定符，也就是 `typeid(const T)== typeid(T))`。

## 使用 `decltype` 说明符

为了用统一方法解决上述问题，C++11标准引入了`decltype`说明 符，使用`decltype`说明符可以获取对象或者表达式的类型，其语法与`typeof`类似：

```cpp
int x1 = 0;
decltype(x1) x2 = 0;
std::cout << typeid(x2).name() << std::endl; // x2 的类型为 int

double x3 = 0;
decltype(x1 + x3) x4 = x1 + x3;
std::cout << typeid(x4).name() << std::endl // x1 + x3的类型为 double

decltype({1, 2}) x5;     // 编译失败，{1, 2} 不是表达式
```

以上代码展示了 `decltype` 的一般用法，代码中分别获取变量 `x1` 和表达式 `x1+x3` 的类型并且声明该类型的变量。 `decltype` 可以在非静态成员变量中使用。

```cpp
struct S1 {
	int x1;
	decltype(x1) x2;
	double x3;
	decltype(x2 + x3) x4;
};
```

比如，在函数形参列表中使用：

```cpp
int x1 = 0;
decltype(x1) sum(decltype(x1) a1, decltype(a1) a2) {
	return a1 + a2;
}

auto x2 = sum(5, 10);
```

为了更好地讨论decltype的优势，需要用到函数返回类型后置的例子

```cpp
auto sum(int a1, int a2) -> int {
	return a1 + a2;
}
```

以上代码以C++11为标准，该标准中auto作为占位符并不能使编译器对函数返回类型进行推导，必须使用返回类型后置的形式指定返回类型。如果接下来想泛化这个函数，让其支持各种类型运算应该怎么办?由于形参不能声明为auto，因此我们需要用到函数模板：

```cpp
template<class T>
T sum(T a1, T a2) {
	return a1 + a2;
}

auto x1 = sum(5, 10);
```

代码看上去很好，但是并不能适应所有情况，因为调用者如果传递不同类型的实参，则无法编译通过：

```cpp
auto x2 = sum(5, 10.5);  // 编译失败，无法确定 T 的类型
```

既然如此，我们只能边写一个更加灵活的函数模版：

```cpp
template<class R, class T1, class T2>
R sum(T1 a1, T2 a2) {
	return a1 + a2;
}

auto x3 = sum<double>(5, 10.5);
```

美中不 足的是我们必须为函数模板指定返回值类型。为了让编译期完成所有 的类型推导工作，我们决定继续优化函数模板：

```cpp
template<class T1, class T2>
auto sum(T1 a1, T2 a2) -> decltype(a1 + a2) {
	return a1 + a2;
}

auto x4 = sum(5, 10.5);
```

auto是返回类型的占位符，参 数类型分别是T1和T2，我们利用decltype说明符能推断表达式的类 型特性，在函数尾部对auto的类型进行说明，如此一来，在实例化 sum函数的时候，编译器就能够知道sum的返回类型了。

上述用法只推荐在C++11标准的编译环境中使用，因为C++14标准 已经支持对auto声明的返回类型进行推导了，所以以上代码可以简化 为:

```cpp
template<class T1, class T2>
auto sum(T1 a1, T2 a2) {
	return a1 + a2;
}

auto x5 = sum(5, 10.5);
```

但是 auto作为返回类型的占位符还存在一些问题，请看下面的例子:

```cpp
template<class T>
auto return_ref(T& t) {
	return t;
}

int x1 = 0;
static_assert(
	std::is_reference_v<decltype(return_ref(x1))>; // 编译错误，返回值不为引用类型
);
```

在上面的代码中，我们期望return_ref返回的是一个T的引用类型，但是如果编译此段代码，则必然会编译失败，因为auto被推导 为值类型，这就是第3章所讲的auto推导规则2。如果想正确地返回引用类型，则需要用到decltype说明符，例如：

```cpp
template<class T>
auto return_ref(T& t) -> decltype(t) {
	return t;
}

int x1 = 0;
static_assert(
	std::is_reference_v<decltype(return_ref(x1))> // 编译成功
);
```

## 推导规则

- 如果e是一个未加括号的标识符表达式(结构化绑定除外)或 者未加括号的类成员访问，则`decltype(e)`推断出的类型是e的类型 T。如果并不存在这样的类型，或者e是一组重载函数，则无法进行推导。

- 如果e是一个函数调用或者仿函数调用，那么 decltype(e) 推断出的类型是其返回值的类型。

- 如果e是一个类型为T的左值，则decltype(e)是T&。

- 如果e是一个类型为T的将亡值，则decltype(e)是T&&。

- 除去以上情况，则decltype(e)是T。

```cpp
const int&& foo();
int i;
struct A {
	double x;
};
const A* a = new A();

decltype(foo());      // decltype(foo())推导类型为 const int&&
decltype(i);          // decltype(i) 推导类型为 int
decltype(a->x);       // decltype(a->x) 推导类型为 double
decltype((a->x));     // decltype((a->x)) 推导类型为 const double& 
```

```cpp
int i;
int *j;
int n[10];
const int&& foo();
decltype(static_cast<short>(i)); // decltype(static_cast<short>(i)) 推导类型为 short

decltype(j);    // decltype(j) 推导类型为 int*
decltype(n);    // decltype(n) 推导类型为 int[10]
decltype(foo);  // decltype(foo) 推导类型为 int const&& (void)

struct A {
	int operator() { return 0; }
};

A a;
decltype(a());  // decltype(a()) 推导类型为 int
```

```cpp
int i;
int *j;
int n[10];
decltype(i=0);     // decltype(i=0) 推导类型为 int&
decltype(0, i);    // decltype(0, i) 推导类型为 int&
decltype(i, 0);    // decltype(i, 0) 推导类型为 int
decltype(n[5]);    // decltype(n[5]) 推导类型为 int&
decltype(*j);      // decltype(*j) 推导类型为 int&
decltype(static_cast<int&&>(i));    // decltype(static_cast<int&&>(i)) 推导类型为 int&&

decltype(i++);     // decltype(i++) 推导类型为 int
decltype(++i);     // decltype(++i) 推导类型为 int&
decltype("hello world");  // const char(&) [12]
```

## cv 限定符的推导

通常情况下，`decltype(e)`所推导的类型会同步e的cv限定符， 比如：

```cpp
const int i = 0;
decltype(i);  // decltype(i) 推导类型为 const int
```

但是还有其他情况，当e是未加括号的成员变量时，父对象表达式的cv限定符会被忽略，不能同步到推导结果：

```cpp
struct A {
	double x;
};
const A* a = new A();
decltype(a->x); // decltype(a->x) 推导类型为 double, const 属性被忽略
```

当然，如果我们给 `a->x` 加上括号，则情况会有所不同。

```cpp
struct A {
	double x;
};
const A* a = new A();
decltype((a->x));   // decltype((a->x)) 推导类型为 const double&
```

## `decltype(auto)`

在C++14标准中出现了`decltype`和`auto`两个关键字的结合体: `decltype(auto)`。它的作用简单来说，就是告诉编译器用` decltype`的推导表达式规则来推导`auto`。另外需要注意的是， `decltype(auto)`必须单独声明，也就是它不能结合指针、引用以及 cv限定符。

```cpp 
int i ;
int&& f();
auto xla = i;            // xla 推导类型为 int
decltype(Auto) xld = il  // xld 推导类型为 int
auto x2a = (i);          // x2a 推导类型为 int
decltype(auto) x2d = (i) // x2d 推导类型为 int&
auto x3a = f();          // x3a 推导类型为 int
decltype(auto) x3d = f(); // x3d 推导类型为 int&&
auto x4a = { 1, 2 };      // x4a 推导类型为 std::initializer_list<int>
decltype(auto) x4d = { 1, 2 }; // 编译失败，{1, 2} 不是表达式
auto *x5a = &i;           // x5a 推导类型为 int*
decltype(auto) *x5d = &i; // 编译失败，decltype(auto)必须单独声明
```

return_ref想返回一个引用类型，但是如果直接使用`auto`，则一定会返回一个值类型。这让我们不得不采用返回类型后置的方式声明返回类型。现在有了`decltype(auto)`组合，我们可以进一步简化代码，消除返回类型后置的语法，例如:

```cpp
template<class T>
decltype(auto) return_ref(T& t) {
	return t;
}

int x1 = 0;
static_assert(
	std::is_reference_v<decltype(return_ref(x1))> // 编译成功
)
```

## `decltype(auto)` 作为非类型模版形参占位符

与auto一样，在C++17标准中`decltype(auto)`也能作为非类型 模板形参的占位符，其推导规则和上面介绍的保持一致。

```cpp
#include <iostream>
template<decltype(auto) N> 
void f() {
	std::cout << N << std::endl;
}

static const int x = 11;
static int y = 7;

int main() {
	f<x>();   // N 为 const int类型
	f<(x)>;   // N 为 const int&类型
	f<y>();   // 编译错误
	f<(y)>(); // N 为 int& 类型
}
```