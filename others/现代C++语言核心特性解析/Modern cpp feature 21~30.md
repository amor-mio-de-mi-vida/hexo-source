---
date: 2024-11-27 16:21:52
date modified: 2024-12-06 13:51:38
title: Modern cpp feature 21~30
tags:
  - cpp
categories:
  - cpp
date created: 2024-09-25 13:26:08
---
Reference: 《现代C++语言核心特性解析》

21. `noexcept`关键字

22. 类型别名和别名模板

23. 指针字面量`nullptr`

24. 三向比较

25. 线程局部存储

26. 扩展的`inline`说明符

27. 常量表达式

28. 确定的表达式求值顺序

29. 字面量优化

30. `alignas` 和 `alignof`

<!-- more -->

# `noexcept`关键字（C++11 C++17 C++20）

## 使用`noexcept`代替`throw`

使用throw声明函数是否抛出异常一直没有什么问题，直到C++11标准引入了移动构造函数。移动构造函数中包含着一个严重的异常陷阱。

当我们想将一个容器的元素移动到另外一个新的容器中时。在C++11之前，由于没有移动语义，我们只能将原始容器的数据复制到新容器中。如果在数据复制的过程中复制构造函数发生了异常，那么我们可以丢弃新的容器，保留原始的容器。在这个环境中，原始容器的内容不会有任何变化。

但是有了移动语义，原始容器的数据会逐一地移动到新容器中，如果数据移动的途中发生异常，那么原始容器也将无法继续使用，因为已经有一部分数据移动到新的容器中。这里读者可能会有疑问，如果发生异常就做一个反向移动操作，恢复原始容器的内容不就可以了吗？实际上，这样做并不可靠，因为我们无法保证恢复的过程中不会抛出异常。

`noexcept`是一个与异常相关的关键字，它既是一个说明符，也是一个运算符。作为说明符，它能够用来说明函数是否会抛出异常，例如：

```cpp
struct X {
	int f() const noexcept {
		return 58;
	}
	void g() noexcept {}
};

int foo() noexcept {
	return 42;
}
```

请注意，`noexcept`只是告诉编译器不会抛出异常，但函数不一定真的不会抛出异常。这相当于对编译器的一种承诺，当我们在声明了`noexcept`的函数中抛出异常时，程序会调用`std::terminate`去结束程序的生命周期。

另外，`noexcept`还能接受一个返回布尔的常量表达式，当表达式评估为`true`的时候，其行为和不带参数一样，表示函数不会抛出异常。反之，当表达式评估为`false`的时候，则表示该函数有可能会抛出异常。这个特性广泛应用于模板当中，例如：

```cpp
template <class T>
T copy(const T& o) noexcept {
	...
}
```

以上代码想实现一个复制函数，并且希望使用`noexcept`优化不抛出异常时的代码。但问题是如果`T`是一个复杂类型，那么调用其复制构造函数是有可能发生异常的。直接声明`noexcept`会导致当函数遇到异常的时候程序被终止，而不给我们处理异常的机会。我们希望只有在`T`是一个基础类型时复制函数才会被声明为`noexcept`，因为基础类型的复制是不会发生异常的。这时就需要用到带参数的`noexcept`了：

```cpp 
template <class T>
T copy(const T &o) noexcept(std::is_fundamental<T>::value) {
	...
}
```

请注意，由于`noexcept`对表达式的评估是在编译阶段执行的，因此表达式必须是一个常量表达式。

实际上，这段代码并不是最好的解决方案，因为我还希望在类型`T`的复制构造函数保证不抛出异常的情况下都使用`noexcept`声明。基于这点考虑，C++标准委员会又赋予了`noexcept`作为运算符的特性。`noexcept`运算符接受表达式参数并返回`true`或`false`。因为该过程是在编译阶段进行，所以表达式本身并不会被执行。而表达式的结果取决于编译器是否在表达式中找到潜在异常：

```cpp
#include <iostream>
int foo() noexcept {
	return 42;
}

int foo1() {
	return 42;
}

int foo2() throw() {
	return 42;
}

int main() {
	std::cout << std::boolalpha;
	std::cout << "noexcept(foo()) = " << noexcept(foo()) << std::endl;
	std::cout << "noexcept(foo1()) = " << noexcept(foo1()) << std::endl;
	std::cout << "noexcept(foo2()) = " << noexcept(foo2()) << std::endl;
}
```

上面这段代码的运行结果如下：

```txt
noexcept(foo()) = true
noexcept(foo1()) = false 
noexcept(foo2()) = true
```

`noexcept`运算符能够准确地判断函数是否有声明不会抛出异常。有了这个工具，我们可以进一步优化复制函数模板：

```cpp
template <class T>
T copy(const T &o) noexcept(noexcept(T(o))) {
	...
}
```

这段代码看起来有些奇怪，因为函数声明中连续出现了两个`noexcept`关键字，只不过两个关键字发挥了不同的作用。其中第二个关键字是运算符，它判断`T(o)`是否有可能抛出异常。而第一个`noexcept`关键字则是说明符，它接受第二个运算符的返回值，以此决定T类型的复制函数是否声明为不抛出异常。

## 用`noexcept`来解决移动构造问题

上文曾提到过，异常的存在对容器数据的移动构成了威胁，因为我们无法保证在移动构造的时候不抛出异常。现在`noexcept`运算符可以判断目标类型的移动构造函数是否有可能抛出异常。如果没有抛出异常的可能，那么函数可以选择进行移动操作；否则将使用传统的复制操作。

```cpp
template<class T>
void swap(T& a, T& b)
noexcept(noexcept(T(std::move(a))) && noexcept(a.operator=(std::move(b)))) {
	T tmp(std::move(a));
	a = std::move(b);
	b = std::move(tmp);
}
```

上面这段代码只做了两件事情：第一，检查类型T的移动构造函数和移动赋值函数是否都不会抛出异常；第二，通过移动构造函数和移动赋值函数移动对象`a`和`b`。在这个函数中使用`noexcept`的好处在于，它让编译器可以根据类型移动函数是否抛出异常来选择不同的优化策略。但是这个函数并没有解决上面容器移动的问题。

```cpp
template<class T>
void swap(T& a, T& b) 
noexcept(noexcept(T(std::move(a))) && noexcept(a.operator=(std::move(b)))) {
	static_assert(noexcept(T(std::move(a))) && noexcept(a.operator=(std::move(b))));
	T tmp(std::move(a));
	a = std::move(b);
	b = std::move(tmp);
}
```

改进版的`swap`在函数内部使用`static_assert`对类型T的移动构造函数和移动赋值函数进行检查，如果其中任何一个抛出异常，那么函数会编译失败。使用这种方法可以迫使类型`T`实现不抛出异常的移动构造函数和移动赋值函数。但是这种实现方式过于强势，我们希望在不满足移动要求的时候，有选择地使用复制方法完成移动操作。

```cpp
#include <iostream>
#include <type_traits>
struct X {
	X() {}
	X(X&&) noexcept {}
	X(const X&) {}
	X operator= (X&&) noexcept { return *this; }
	X operator= (const X&) { return *this; }
};

struct X1 {
	X1() {}
	X1(X1&&) {}
	X1(const X1&) {}
	X1 operator= (X1&&) { return *this; }
	X1 operator= (const X1&) { return *this; }
};

template<typename T>
void swap_impl(T& a, T& b, std::integral_constant<bool, true>)
noexcept {
	T tmp(std::move(a));
	a = std::move(b);
	b = std::move(tmp);
}

template<typename T>
void swap_impl(T& a, T& b, std::integral_constant<bool, false>) {
	T tmp(a);
	a = b;
	b = tmp;
}

template<typename T>
void swap(T& a, T& b) 
noexcept(noexcept(swap_impl(a, b, std::integral_constant<bool, noexcept(T(std::move(a))) && noexcept(a.operator=(std::move(b)))> ()))) {
	swap_impl(a, b, std::integral_constant<bool, noexcept(T(std::move(a))) && noexcept(a.operator=(std::move(b))) > ());
}

int main() {
	X x1, x2;
	swap(x1, x2);
	
	X1 x3, x4;
	swap(x3, x4);
}
```

以上代码实现了两个版本的swap_impl，它们的形参列表的前两个形参是相同的，只有第三个形参类型不同。第三个形参为`std::integral_constant<bool, true>`的函数会使用移动的方法交换数据，而第三个参数为`std::integral_ constant<bool,false>`的函数则会使用复制的方法来交换数据。`swap`函数会调用`swap_impl`，并且以移动构造函数和移动赋值函数是否会抛出异常为模板实参来实例化`swap_impl`的第三个参数。这样，不抛出异常的类型会实例化一个类型为`std::integral_constant<bool, true>`的对象，并调用使用移动方法的`swap_impl`；反之则调用使用复制方法的`swap_impl`。

## `noexcept`和`throw()`

在了解了`noexcept`以后，现在是时候对比一下`noexcept`和`throw()`两种方法了。请注意，这两种指明不抛出异常的方法在外在行为上是一样的。如果用`noexcept`运算符去探测`noexcept`和`throw()`声明的函数，会返回相同的结果。

但实际上在C++11标准中，它们在实现上确实是有一些差异的。如果一个函数在声明了`noexcept`的基础上抛出了异常，那么程序将不需要展开堆栈，并且它可以随时停止展开。另外，它不会调用`std::unexpected`，而是调用`std::terminate`结束程序。而`throw()`则需要展开堆栈，并调用`std::unexpected`。这些差异让使用`noexcept`程序拥有更高的性能。在C++17标准中，`throw()`成为`noexcept`的一个别名，也就是说`throw()`和`noexcept`拥有了同样的行为和实现。另外，在C++17标准中只有`throw()`被保留了下来，其他用`throw`声明函数抛出异常的方法都被移除了。在C++20中`throw()`也被标准移除了，使用`throw`声明函数异常的方法正式退出了历史舞台。

## 默认使用noexcept的函数

**默认构造函数、默认复制构造函数、默认赋值函数、默认移动构造函数和默认移动赋值函数**。有一个额外要求，对应的函数在类型的基类和成员中也具有`noexcept`声明，否则其对应函数将不再默认带有`noexcept`声明。另外，自定义实现的函数默认也不会带有`noexcept`声明：

```cpp
#include <iostream>

struct X {
};

#define PRINT_NOEXCEPT(x) std::cout << #x << " = " << x << std::endl

int main() {
	X x;
	std::cout << std::boolalpha;
	PRINT_NOEXCEPT(noexcept(X()));
	PRINT_NOEXCEPT(noexcept(X(x)));
	PRINT_NOEXCEPT(noexcept(X(std::move(x))));
	PRINT_NOEXCEPT(noexcept(x.operator=(x)));
	PRINT_NOEXCEPT(noexcept(x.operator=(std::move(x))));
}
```

以上代码的运行输出结果如下：

```txt
noexcept(X()) = true
noexcept(X(x)) = true
noexept(X(std::move(x))) = true
noexcept(x.operator=(x)) = true
noexcept(x.operator=(std::move(x))) = true
```

可以看到编译器默认实现的这些函数都是带有`noexcept`声明的。如果我们在类型X中加入某个成员变量M，情况会根据M的具体实现发生变化：

```cpp
#include <iostream>

struct M {
	M() {}
	M(const M&) {}
	M(M&&) noexcept {}
	M operator= (const M&) noexcept { return *this; }
	M operator= (M&&) { return *this; }
};

struct X {
	M m;
};

#define PRINT_NOEXCEPT(x) std::cout << #x << " = " << x << std::endl

int main() {
	X x;
	std::cout << std::boolalpha;
	PRINT_NOEXCEPT(noexcept(X()));
	PRINT_NOEXCEPT(noexcept(X(x)));
	PRINT_NOEXCEPT(noexcept(X(std::move(x))));
	PRINT_NOEXCEPT(noexcept(x.operator=(x)));
	PRINT_NOEXCEPT(noexcept(x.operator=(std::move(x))));
}
```

这时的结果如下：

```txt
noexcept(X()) = false
noexcept(X(x)) = false
noexcept(X(std::move(x))) = true
noexcept(x.operator=(x)) = true
noexcept(x.operator=(std::move(x))) = false 
```


**类型的析构函数以及`delete`运算符默认带有`noexcept`声明**，请注意即使自定义实现的析构函数也会默认带有`noexcept`声明，除非类型本身或者其基类和成员明确使用`noexcept(false)`声明析构函数，以上也同样适用于`delete`运算符：

```cpp
#include <iostream>

struct M {
	~M() noexcept(false) {}
};

struct X {
};

struct X1 {
	~X1() {}
};

struct X2 {
	~X2() noexcept(false0) {}
};

struct X3 {
	M m;
};

#define PRINT_NOEXCEPT(x) std::cout << #x << " = " << x << std::endl

int main() {
	X *x = new X;
	X1 *x1 = new X1;
	X2 *x2 = new X2;
	X3 *x3 = new X3;
	std::cout << std::boolalpha;
	PRINT_NOEXCEPT(noexcept(x->~X()));
	PRINT_NOEXCEPT(noexcept(x1->~X1()));
	PRINT_NOEXCEPT(noexcept(x2->~X2()));
	PRINT_NOEXCEPT(noexcept(x3->~X3()));
	PRINT_NOEXCEPT(noexcept(delete x));
	PRINT_NOEXCEPT(noexcept(delete x1));
	PRINT_NOEXCEPT(noexcept(delete x2));
	PRINT_NOEXCEPT(noexcept(delete x3));
}
```

以上代码的运行输出结果如下：

```txt
noexcept(x->X()) = true
noexcept(x1->~X1()) = true
noexcept(x2->~X2()) = false
noexcept(x3->~X3()) = false 
noexcept(delete x) = true
noexcept(delete x1) = true
noexcept(delete x2) = false 
noexcept(delete x3) = false
```

## 使用noexcept的时机

那么哪些函数可以使用`noexcept`声明呢？这里总结了两种情况。

- **一定不会出现异常的函数。通常情况下，这种函数非常简短，例如求一个整数的绝对值、对基本类型的初始化等。**

- **当我们的目标是提供不会失败或者不会抛出异常的函数时可以使用`noexcept`声明**。对于保证不会失败的函数，例如内存释放函数，一旦出现异常，相对于捕获和处理异常，终止程序是一种更好的选择。这也是`delete`会默认带有`noexcept`声明的原因。另外，对于保证不会抛出异常的函数而言，即使有错误发生，函数也更倾向用返回错误码的方式而不是抛出异常。

除了上述两种理由，我认为保持函数的异常中立是一个明智的选择，因为将函数从没有`noexcept`声明修改为带`noexcept`声明并不会付出额外代价，而反过来的代价有可能是很大的。

## 将异常规范作为类型的一部分

在C++17标准之前，异常规范没有作为类型系统的一部分，所以下面的代码在编译阶段不会出现问题：

```cpp
void (*fp) () noexcept = nullptr;
void foo() {}

int main() {
	fp = &foo;
}
```

在C++17之前，它们的类型是相同的，也就是说`std::is_same <decltype(fp), decltype(&foo)>::value`返回的结果为`true`。显然，这种宽松的规则会带来一些问题，例如一个会抛出异常的函数通过一个保证不抛出异常的函数指针进行调用，结果该函数确实抛出了异常，正常流程本应该是由程序捕获异常并进行下一步处理，但是由于函数指针保证不会抛出异常，因此程序直接调用`std::terminate`函数中止了程序：

```cpp
#include <iostream>
#include <string>

void(*fp)() noexcept = nullptr;
void foo() {
	throw(5);
}

int main() {
	fp = &foo;
	try {
		fp();
	} catch (int e) {
		std::cout << e << std::endl;
	}
}
```

为了解决此类问题，C++17标准将异常规范引入了类型系统。这样一来，`fp = &foo`就无法通过编译了，因为`fp`和`&foo`变成了不同的类型，`std::is_same <decltype(fp),decltype(&foo)>::value`会返回false。值得注意的是，虽然类型系统引入异常规范导致`noexcept`声明的函数指针无法接受没有`noexcept`声明的函数，但是反过来却是被允许的，比如：

```cpp
void(*fp)() = nullptr;
void foo() noexcept {}

int main() {
	fp = &foo;
}
```

这里的原因很容易理解，一方面这个设定可以保证现有代码的兼容性，旧代码不会因为没有声明`noexcept`的函数指针而编译报错。另一方面，在语义上也是可以接受的，因为函数指针既没有保证会抛出异常，也没有保证不会抛出异常，所以接受一个保证不会抛出异常的函数也合情合理。同样，虚函数的重写也遵守这个规则，例如：

```cpp
class Base {
public:
	virtual void foo() noexcept {}
};
class Derived : public Base {
public:
	void foo() override {};
};
```

以上代码无法编译成功，因为派生类试图用没有声明`noexcept`的虚函数重写基类中声明`noexcept`的虚函数，这是不允许的。但反过来是可以通过编译的：

```cpp
class Base {
public:
	virtual void foo() {}
};

class Derived : public Base {
public:
	void foo() noexcept override {};
};
```

最后需要注意的是模板带来的兼容性问题，在标准文档中给出了这样一个例子：

```cpp
void g1() noexcept {}
void g2() {}
template<class T> void f(T*, T*) {}

int main() {
	f(g1, g2);
}
```

在C++17中g1和g2已经是不同类型的函数，编译器无法推导出同一个模板参数，导致编译失败。为了让这段编译成功，需要简单修改一下函数模板：

```cpp
template<class T1, class T2> void f(T1*, T2*) {}
```

# 类型别名和别名模板（C++11 C++14）

## 类型别名

在C++的程序中，我们经常会看到特别长的类型名，比如`std::map<int, std::string>::const_iterator`。为了让代码看起来更加简洁，往往会使用`typedef`为较长的类型名定义一个别名，例如：

```cpp
typedef std::map<int, std::string>::const_iterator map_const_iter;
map_const_iter iter;
```

C++11标准提供了一个新的定义类型别名的方法，该方法使用`using`关键字，具体语法如下：

```cpp
using identifier = type-id
```

其中`identifier`是类型的别名标识符，`type-id`是已有的类型名。相对于`typedef`，我更喜欢`using`的语法，因为它很像是一个赋值表达式，只不过它所“赋值”的是一个类型。这种表达式在定义函数指针类型的别名时显得格外清晰：

```cpp
typedef void (*func1)(int, int);
using func2 = void(*)(int, int);
```

## 别名模板

前面我们已经了解到使用`using`定义别名的基本用法，但是显然C++委员会不会因为这点内容就添加一个新的关键字。事实上`using`还承担着一个更加重要的特性——别名模板。所谓别名模板本质上也应该是一种模板，它的实例化过程是用自己的模板参数替换原始模板的模板参数，并实例化原始模板。定义别名模板的语法和定义类型别名并没有太大差异，只是多了模板形参列表：

```cpp
template <template-parameter-list>
using identifier = type-id;
```

其中`template-parameter-list`是模板的形参列表，而`identifier`和`type-id`是别名类模板型名和原始类模板型名。下面来看一个例子：

```cpp
#include <map>
#include <string>

template<class T>
using int_map = std::map<int, T>

int main() {
	int_map<std::string> int2string;
	int2string[11] = "7";
}
```

在上面的代码中，`int_map`是一个别名模板，它有一个模板形参。当`int_map`发生实例化的时候，模板的实参`std::string`会替换`std::map<int, T>`中的`T`，所以真正实例化的类型是`std::map<int, std::string>`。通过这种方式，我们可以在模板形参比较多的时候简化模板形参。

看到这里，有模板元编程经验的读者可能会提出`typedef`其实也能做到相同的事情。没错，我们是可以用`typedef`来改写上面的代码：

```cpp
#include <map>
#include <string>
template<class T>
struct int_map {
	typedef std::map<int, T> type;
};

int main() {
	int_map<std::string>::type int2string;
	int2string[11] = "7";
}
```

以上代码使用`typedef`和类型嵌套的方案也能达到同样的目的。不过很明显这种方案要复杂不少，不仅要定义一个`int_map`的结构体类型，还需要在类型里使用`typedef`来定义目标类型，最后必须使用`int_map<std::string>::type`来声明变量。除此之外，如果遇上了待决的类型，还需要在变量声明前加上`typename`关键字：

```cpp
template<class T>
struct int_map {
	typedef std::map<int, T> type;
};

template<class T>
struct X {
	typename int_map<T>::type int2other; // 必须带有typename关键字，否则编译错误
};
```

在上面这段代码中，类模板`X`没有确定模板形参T的类型，所以`int_map<T>::type`是一个未决类型，也就是说`int_map<T>::type`既有可能是一个类型，也有可能是一个静态成员变量，编译器是无法处理这种情况的。这里的`typename`关键字告诉编译器应该将`int_map<T>::type`作为类型来处理。而别名模板不会有`::type`的困扰，当然也不会有这样的问题了：

```cpp
template<class T>
using int_map = std::map<int, T>;

template<class T>
struct X {
	int_map<T> int2other; // 编译成功，别名模板不会有任何问题
};
```

值得一提的是，虽然别名模板有很多`typedef`不具备的优势，但是C++11标准库中的模板元编程函数都还是使用的`typedef`和类型嵌套的方案，例如：

```cpp
template<bool, typename _Tp = void>
struct enable_if { };

template<typename _Tp>
struct enable_if<true, _Tp>
{ typedef _Tp type; }
```

不过这种情况在C++14中得到了改善，在C++14标准库中模板元编程函数已经有了别名模板的版本。当然，为了保证与老代码的兼容性，`typedef`的方案依然存在。别名模板的模板元编程函数使用`_t`作为其名称的后缀以示区分：

```cpp
template<bool, _Cond, typename _Tp = void>
using enable_if_t = typename enable_if<_Cond, _Tp>::type;
```

# 指针字面量`nullptr`（C++11）

## 零值整数字面量

在C++标准中有一条特殊的规则，即0既是一个整型常量，又是一个空指针常量。0作为空指针常量还能隐式地转换为各种指针类型。比如我们在初始化变量的时候经常看到的代码：

```cpp
char* p = NULL;
int x = 0;
```

这里的NULL是一个宏，在C++11标准之前其本质就是0：

```cpp
#ifndef NULL
	#ifdef __cplusplus
		#define NULL 0
	#else 
		#define NULL ((void*)0)
	#endif
#endif
```

使用0代表不同类型的特殊规则给C++带来了二义性，对C++的学习和使用造成了不小的麻烦，下面是C++标准文档的两个例子：

```cpp
// 例子1
void f(int) {
	std::cout << "int" << std::endl;
}

void f(char*) {
	std::cout << "char*" << std::endl;
}

f(NULL);
f(reinterpret_cast<char *> (NULL));
```

在上面这段代码中`f(NULL)`函数调用的是`f(int)`函数，因为`NULL`会被优先解析为整数类型。没有办法让编译器自动识别传入`NULL`的意图，除非使用类型转换，将`NULL`转换到`char*`，`f(reinterpret_cast<char *>(NULL))`可以正确地调用`f(char*)`函数。注意，上面的代码可以在MSVC中编译执行。在GCC中，我们会得到一个NULL有二义性的错误提示。

下面这个例子看起来就更加奇怪了：

```cpp
// 例子2
std::string s1(false);
std::string s2(true);
```

以上代码可以用MSVC编译，其中`s1`可以成功编译，但是`s2`则会编译失败。原因是`false`被隐式转换为`0`，而`0`又能作为空指针常量转换为`const char * const`，所以`s1`可以编译成功，`true`则没有这样的待遇。在GCC中，编译器对这种代码也进行了特殊处理，如果用C++11(-std=c++11)及其之后的标准来编译，则两条代码均会报错。但是如果用C++03以及之前的标准来编译，则虽然第一句代码能编译通过，但会给出警告信息，第二句代码依然编译失败。

## `nullptr`关键字

鉴于`0`作为空指针常量的种种劣势，C++标准委员会在C++11中添加关键字`nullptr`表示空指针的字面量，它是一个`std::nullptr_t`类型的纯右值。`nullptr`的用途非常单纯，就是用来指示空指针，它不允许运用在算术表达式中或者与非指针类型进行比较（除了空指针常量`0`）。它还可以隐式转换为各种指针类型，但是无法隐式转换到非指针类型。注意，`0`依然保留着可以代表整数和空指针常量的特殊能力，保留这一点是为了让C++11标准兼容以前的C++代码。所以，下面给出的例子都能够顺利地通过编译：

```cpp
char* ch = nullptr;
char* ch2 = 0;
assert(ch == 0);
assert(ch == nullptr);
assert(!ch);
assert(ch2 == nullptr);
assert(nullptr == 0);
```

将指针变量初始化为`0`或者`nullptr`的效果是一样的，在初始化以后它们也能够与`0`或者`nullptr`进行比较。从最后一句代码看出`nullptr`也可以和`0`直接比较，返回值为`true`。虽然`nullptr`可以和`0`进行比较，但这并不代表它的类型为整型，同时它也不能隐式转换为整型：

```cpp
int n1 = nullptr;
char* ch1 = true ? 0 : nullptr;
int n2 = true ? nullptr : nullptr;
int n3 = true ? 0 : nullptr;
```

以上代码的第一句和第三句操作都是将一个`std::nullptr_t`类型赋值到int类型变量。由于这个转换并不能自动进行，因此会产生编译错误。而第二句和第四句中，因为条件表达式的 :前后类型不一致，而且无法简单扩展类型，所以同样会产生编译错误。请注意，上面代码中的第二句在MSVC中是可以编译通过的。

```cpp
namespace std {
	using nullptr_t = decltype(nullptr);
	// 等价于
	typedef decltype(nullptr nullptr_t);
}
static_assert(sizeof(std::nullptr_t) == sizeof(void*));
```

我们还可以使用`std::nullptr_t`去创建自己的`nullptr`，并且有与`nullptr`相同的功能：

```cpp
std::nullptr_t null1, null2;

char* ch = null1;
char* ch2 = null2;
assert(ch == 0);
assert(ch == nullptr);
assert(ch == null2);
assert(null1 == null2);
assert(nullptr == null1);
```

虽然这段代码中`null1`、`null2`和`nullptr`的能力相同，但是它们还是有很大区别的。首先，`nullptr`是关键字，而其他两个是声明的变量。其次，`nullptr`是一个纯右值，而其他两个是左值：

```cpp
std::nullptr_t null1, null2;
std::cout << "&null1 = " << &null1 << std::endl; // null1和null2是左值，可以成功获取对象指针，
std::cout << "&null2 = " << &null2 << std::endl; // 并且指针指向的内存地址不同
```

上面这段代码对`null1`和`null2`做了取地址的操作，并且返回不同的内存地址，证明它们都是左值。但是这个操作用在`nullptr`上肯定会产生编译错误：

```cpp
std::cout << "&nullptr = " << &nullptr << std::endl; // 编译失败，取地址操作需要一个左值
```

`nullptr`是一个纯右值，对`nullptr`进行取地址操作就如同对常数取地址一样，这显然是错误的。讨论过`nullptr`的特性以后，我们再来看一看重载函数的例子：

```cpp
void f(int) {
	std::cout << "int" << std::endl;
}

void f(char*) {
	std::cout << "char*" << std::endl;
}

f(nullptr);
```

使用`nullptr`的另一个好处是，我们可以为函数模板或者类设计一些空指针类型的特化版本。在C++11以前这是不可能实现的，因为`0`的推导类型是`int`而不是空指针类型。现在我们可以利用`nullptr`的类型为`std::nullptr_t`写出下面的代码：

```cpp
#include <iostream>

template<class T>
struct widget {
	widget() {
		std::cout << "template" << std::endl;
	}
};

template<>
struct widget<std::nullptr_t> {
	widget() {
		std::cout << "nullptr" << std::endl;
	}
};

template<class T>
widget<T>* make_widget(T) {
	return new widget<T>();
}

int main() {
	auto w1 = make_widget(0);
	auto w2 = make_widget(nullptr);
}
```

# 三向比较（C++20）

## “太空飞船”（spaceship）运算符

三向比较就是在形如`lhs <=> rhs`的表达式中，两个比较的操作数`lhs`和`rhs`通过`<=>`比较可能产生3种结果，该结果可以和0比较，小于0、等于0或者大于0分别对应`lhs < rhs`、`lhs == rhs`和`lhs > rhs`。举例来说：

```cpp
bool b = 7 <=> 11 < 0; // b == true
```

请注意，运算符`<=>`的返回值只能与0和自身类型来比较，如果同其他数值比较，编译器会报错：

```cpp
bool b = 7 <=> 11 < 100; // 编译失败，<=>的结果不能与除0以外的数值比较
```

## 三向比较的返回类型

可以看出<=>的返回结果并不是一个普通类型，根据标准三向比较会返回3种类型，分别为`std::strong_ordering`、`std::weak_ordering`以及`std:: partial_ordering`，而这3种类型又会分为有3～4种最终结果，下面就来一一介绍它们。

### `std::strong_ordering`

`std::strong_ordering`类型有3种比较结果，分别为`std::strong_ ordering::less`、`std::strong_ordering::equal`以及`std::strong_ordering::greater`。表达式`lhs <=> rhs`分别表示`lhs <rhs、lhs == rhs`以及`lhs > rhs。std::strong_ordering`类型的结果强调的是strong的含义，表达的是一种可替换性，简单来说，若`lhs == rhs`，那么在任何情况下rhs和lhs都可以相互替换，也就是`fx(lhs) == fx(rhs)`。

```cpp
std::cout << typeid(decltype(7<=>11)).name();
```

对于有复杂结构的类型，`std::strong_ordering`要求其数据成员和基类的三向比较结果都为`std::strong_ordering`。例如：

```cpp
#include <compare>

struct B {
	int a;
	long b;
	auto operator <=> (const B&) const = default;
};

struct D : B {
	short c;
	auto operator <=> (const D&) const = default;
};

D x1, x2;
std::cout << typeid(decltype(x1 <=> x2)).name();
```

请注意，默认情况下自定义类型是不存在三向比较运算符函数的，需要用户显式默认声明，比如在结构体B和D中声明`auto operator <=> (const B&) const =default;`和`auto operator <=> (const D&) const =default;`。对结构体B而言，由于`int`和`long`的比较结果都是`std::strong_ordering`，因此结构体B的三向比较结果也是`std::strong_ordering`。同理，对于结构体D，其基类和成员的比较结果是`std::strong_ordering`，D的三向比较结果同样是`std::strong_ordering`。另外，明确运算符的返回类型，使用`std::strong_ ordering`替换`auto`也是没问题的。

### `std::weak_ordering`

`std::weak_ordering`类型也有3种比较结果，分别为`std::weak_ ordering::less`、`std::weak_ordering::equivalent`以及`std::weak_ordering::greater`。`std::weak_ordering`的含义正好与`std::strong_ ordering`相对，表达的是不可替换性。即若有`lhs== rhs`，则`rhs`和`lhs`不可以相互替换，也就是`fx(lhs) != fx(rhs)`。这种情况在基础类型中并没有，但是它常常发生在用户自定义类中，比如一个大小写不敏感的字符串类：

```cpp
#include <compare>
#include <string>

int ci_compare(const char* s1, const char* s2) {
	while (tolower(*s1) == tolower(*s2++)) {
		if (*s1++ == '\0') {
			return 0;
		}
	}
	return tolower(*s1) - tolower(*--s2);
}

class CIString {
public:
	CIString(const char* s) : str_(s) {}
	std::weak_ordering operator<=>(const CIString& b) const {
		return ci_compare(str_.c_str(), b.str_.c_str()) <=> 0;
	}
private:
	std::string str_;
};

CIString s1{ "HELLO" }, s2{ "hello" };
std::cout << (s1 <=> s2 == 0); // 输出为true
```

以上代码实现了一个简单的大小写不敏感的字符串类，它对于s1和s2的比较结果是`std::weak_ordering::equivalent`，表示两个操作数是等价的，但是它们不是相等的也不能相互替换。当`std::weak_ordering`和`std::strong_ ordering`同时出现在基类和数据成员的类型中时，该类型的三向比较结果是`std::weak_ordering`，例如：

```cpp
struct D : B {
	CIString c{""}
	auto operator <=> (const D&) const = default;
};

D w1, w2;
std::cout << typeid(decltype(w1<=>w2)).name();
```

用MSVC编译运行上面这段代码会输出`class std::weak_ordering`，因为D中的数据成员`CIString`的三向比较结果为`std::weak_ordering`。请注意，如果显式声明默认三向比较运算符函数为`std::strong_ordering operator <=> (const D&) const = default;`，那么一定会遭遇到一个编译错误

### `std::partial_ordering`

`std::partial_ordering`约束力比`std::weak_ordering`更弱，它可以接受当`lhs == rhs`时rhs和lhs不能相互替换，同时它还能给出第四个结果`std::partial_ ordering::unordered`，表示进行比较的两个操作数没有关系。比如基础类型中的浮点数：

```cpp
std::cout << typeid(decltype(7.7<=>11.1)).name();
```

用MSVC编译运行以上代码会输出`class std::partial_ordering`。之所以会输出`class std::partial_ordering`而不是`std::strong_ordering`，是因为浮点的集合中存在一个特殊的`NaN`，它和其他浮点数值是没关系的：

```cpp
std::cout << ((0.0 / 0.0 <=> 1.0) == std::partial_ordering::unordered);
```

这段代码编译输出的结果为true。当`std::weak_ordering`和 `std:: partial_ordering`同时出现在基类和数据成员的类型中时，该类型的三向比较结果是`std::partial_ordering`，例如：

```cpp
struct D : B {
	CIString c{""};
	float u;
	auto operator <=> (const D&) const = default;
};

D w1, s2;
std::cout << typeid(decltype(w1 <=> w2)).name();
```

用MSVC编译运行以上代码会输出`class std::partial_ordering`，因为`D`中的数据成员`u`的三向比较结果为`std::partial_ordering`，同样，显式声明为其他返回类型也会让编译器报错。在C++20的标准库中有一个模板元函数 `std::common_comparison_category`，它可以帮助我们在一个类型合集中判断出最终三向比较的结果类型，当类型合集中存在不支持三向比较的类型时，该模板元函数返回`void`。

再次强调一下，`std::strong_ordering`、`std::weak_ordering`和`std::partial_ordering`只能与`0`和类型自身比较。深究其原因，是这3个类只实现了参数类型为自身类型和`nullptr_t`的比较运算符函数。

## 对基础类型的支持

**对两个算术类型的操作数进行一般算术转换，然后进行比较。** 其中整型的比较结果为`std::strong_ordering`，浮点型的比较结果为`std::partial_ordering`。例如7 <=> 11.1中，整型7会转换为浮点类型，然后再进行比较，最终结果为`std::partial_ordering`类型。

**对于无作用域枚举类型和整型操作数，枚举类型会转换为整型再进行比较，无作用域枚举类型无法与浮点类型比较**：

```cpp
enum color {
	red
};

auto r = red <=> 11; //编译成功
auto r = red <=> 11.1; //编译失败
```

**对两个相同枚举类型的操作数比较结果，如果枚举类型不同，则无法编译**。

**对于其中一个操作数为bool类型的情况，另一个操作数必须也是bool类型，否则无法编译。** 比较结果为`std::strong_ordering`。

**不支持作比较的两个操作数为数组的情况，会导致编译出错**，例如：

```cpp
int arr1[5];
int arr2[5];
auto r = arr1 <=> arr2; // 编译失败
```

**对于其中一个操作数为指针类型的情况，需要另一个操作数是同样类型的指针，或者是可以转换为相同类型的指针，比如数组到指针的转换、派生类指针到基类指针的转换等**，最终比较结果为`std::strong_ordering`：

```cpp
char arr1[5];
char arr2[5];
char* ptr = arr2;
auto r = ptr <=> arr1;
```

上面的代码可以编译成功，若将代码中的`arr1`改写为`int arr1[5]`，则无法编译，因为`int [5]`无法转换为`char *`。如果将`char * ptr = arr2;`修改为`void * ptr = arr2;`，代码就可以编译成功了

## 自动生成的比较运算符函数

标准库中提供了一个名为`std::rel_ops`的命名空间，在用户自定义类型已经提供了`==`运算符函数和`<`运算符函数的情况下，帮助用户实现其他4种运算符函数，包括`!=`、`>`、`<=`和`>=`，例如：

```cpp
#include <string>
#include <utility>
class CIString2 {
public:
	CIString2(const char* s) : str_(s) {}
	bool operator < (const CIString2& b) const {
		return ci_compare(str_.c_str(), b.str_.c_str()) < 0;
	}
private:
	std::string str_;
};

using namespace std::rel_ops;
CIString2 s1{ "hello" }, s2{ "world" };
bool r = s1 >= s2;
```

不过因为C++20标准有了三向比较运算符的关系，所以不推荐上面这种做法了。C++20标准规定，如果用户为自定义类型声明了三向比较运算符，那么编译器会为其自动生成`<`、`>`、`<=`和`>=`这4种运算符函数。对于`CIString`我们可以直接使用这4种运算符函数：

```cpp
CIString s1{ "hello" }, s2{ "world" };
bool r = s1 >= s2;
``` 

那么这里就会产生一个疑问，很明显三向比较运算符能表达两个操作数是相等或者等价的含义，为什么标准只允许自动生成4种运算符函数，却不能自动生成`==`和`=!`这两个运算符函数呢？实际上这里存在一个严重的性能问题。在C++20标准拟定三向比较的早期，是允许通过三向比较自动生成6个比较运算符函数的，而三向比较的结果类型也不是3种而是5种，多出来的两种分别是`std::strong_ equality`和`std::weak_equality`。但是在提案文档p1190中提出了一个严重的性能问题。简单来说，假设有一个结构体：

```cpp
struct S {
	std::vector<std::string> names;
	auto operator <=> (const S&) const = default;
};
```

它的三向比较运算符的默认实现这样的：

```cpp
template<typename T>
std::strong_ordering operator<=>(const std::vector<T>& lhs, const std::vector<T>& rhs) {
	size_t min_size = min(lhs.size(), rhs.size());
	for (size_t i = 0; i != min_size; i++) {
		if (auto const cmp = std::compare_3way(lhs[i], rhs[i]); cmp != 0) {
			return cmp;
		}
	}
	return lhs.size() <=> rhs.size();
}
```

这个实现对于<和>这样的运算符函数没有问题，因为需要比较容器中的每个元素。但是`==`运算符就显得十分低效，对于`==`运算符高效的做法是先比较容器中的元素数量是否相等，如果元素数量不同，则直接返回false：

```cpp
template<typename T>
bool operator==(const std::vector<T>& lhs, const std::vector<T>& rhs) {
	const size_t size = lhs.size();
	if (size != rhs.size()) {
		return false;
	}
	for (size_t i = 0; i != size; i++) {
		if (lhs[i] != rhs[i]) {
			return false;
		}
	}
	return true;
}
```

想象一下，如果标准允许用三向比较的算法自动生成`==`运算符函数会发生什么事情，很多旧代码升级编译环境后会发现运行效率下降了，尤其是在容器中元素数量众多且每个元素数据量庞大的情况下。很少有程序员会注意到三向比较算法的细节，导致这个性能问题难以排查。基于这种考虑，C++委员会修改了原来的三向比较提案，规定声明三向比较运算符函数只能够自动生成4种比较运算符函数。由于不需要负责判断是否相等，因此`std::strong_equality`和`std::weak_ equality`也退出了历史舞台。对于`==`和`!=`两种比较运算符函数，只需要多声明一个`==`运算符函数，`!=`运算符函数会根据前者自动生成：

```cpp
class CIString {
public:
	CIString(const char* s) : str_(s) {}
	std::week_ordering operator<=>(const CIString& b) const {
		return ci_compare(str_.c_str(), b.str_.c_str()) <=> 0;
	}
	bool operator == (const CIString& b) const {
		return ci_compare(str_.c_str(), b.str_.c_str()) == 0;
	}
private:
	std::string str_;
};

CIString s1{ "hello" }, s2{ "world" };
bool r1 = s2 >= s2; // 调用operator<=>
bool r2 = s1 == s2; // 调用operator ==
```

## 兼容旧代码

现在C++20标准已经推荐使用`<=>`和`==`运算符自动生成其他比较运算符函数，而使用`<`、`==`以及`std::rel_ops`生成其他比较运算符函数则会因为`std::rel_ops`已经不被推荐使用而被编译器警告。那么对于老代码，我们是否需要去实现一套`<=>`和`==`运算符函数呢？其实大可不必，C++委员会在裁决这项修改的时候已经考虑到老代码的维护成本，所以做了兼容性处理，即在用户自定义类型中，实现了`<`、`==`运算符函数的数据成员类型，在该类型的三向比较中将自动生成合适的比较代码。比如：

```cpp
struct Legacy {
	int n;
	bool operator==(const Legacy& rhs) const {
		return n == rhs.n;
	}
	bool operator<(const Legacy& rhs) const {
		return n < rhs.n;
	}
};

struct TreeWay {
	Legacy m;
	std::strong_ordering operator <=> (const TreeWay &) const = default;
};

TreeWay t1, t2; 
bool r = t1 < t2;
```

在上面的代码中，结构体TreeWay的三向比较操作会调用结构体Legacy中的`<`和`==`运算符来完成，其代码类似于：

```cpp
struct TreeWay {
	Legacy m;
	std::strong_ordering operator<=>(const TreeWay& rhs) const {
		if (m < rhs.m) return std::string_ordering::less;
		if (m == rhs.m) return std::strong_ordering::equal;
		return std::strong_ordering::greater;
	}
};
```

需要注意的是，这里`operator<=>`必须显式声明返回类型为`std::strong_ ordering`，使用`auto`是无法通过编译的。

# 线程局部存储（C++11）

## 操作系统和编译器对线程局部存储的支持

在Windows中可以通过调用API函数`TlsAlloc`来分配一个未使用的线程局部存储槽索引`（TLS slot index）`，这个索引实际上是Windows内部线程环境块（TEB）中线程局部存储数组的索引。通过API函数`TlsGetValue`与`TlsSetValue`可以获取和设置线程局部存储数组对应于索引元素的值。API函数`TlsFree`用于释放线程局部存储槽索引。

Linux使用了pthreads（POSIX threads）作为线程接口，在`pthreads`中我们可以调用`pthread_key_create`与`pthread_key_delete`创建与删除一个类型为`pthread_key_t`的键。利用这个键可以使用`pthread_setspecific`函数设置线程相关的内存数据，当然，我们随后还能够通过`pthread_getspecific`函数获取之前设置的内存数据。

在C++11标准确定之前，各个编译器也用了自定义的方法支持线程局部存储。比如gcc和clang添加了关键字`__thread`来声明线程局部存储变量，而Visual Studio C++则是使用`__declspec(thread)`。虽然它们都有各自的方法声明线程局部存储变量，但是其使用范围和规则却存在一些区别，这种情况增加了C++的学习成本，也是C++标准委员会不愿意看到的。于是在C++11标准中正式添加了新的`thread_local`说明符来声明线程局部存储变量。

## `thread_local`说明符

`thread_local`说明符可以用来声明线程生命周期的对象，它能与`static`或`extern`结合，分别指定内部或外部链接，不过额外的`static`并不影响对象的生命周期。换句话说，`static`并不影响其线程局部存储的属性：

```cpp
struct X {
	thread_local static int i;
};

thread_local X a;
 
int main() {
	thread_local X b;
}
```

被`thread_local`声明的变量在行为上非常像静态变量，只不过多了线程属性，当然这也是线程局部存储能出现在我们的视野中的一个关键原因，它能够解决全局变量或者静态变量在多线程操作中存在的问题，一个典型的例子就是`errno`。

`errno`通常用于存储程序当中上一次发生的错误，早期它是一个静态变量，由于当时大多数程序是单线程的，因此没有任何问题。但是到了多线程时代，这种`errno`就不能满足需求了。为了规避由此产生的不确定性，POSIX将`errno`重新定义为线程独立的变量，为了实现这个定义就需要用到线程局部存储，直到C++11之前，`errno`都是一个静态变量，而从C++11开始`errno`被修改为一个线程局部存储变量。

值得注意的是，使用取地址运算符`&`取到的线程局部存储变量的地址是运行时被计算出来的，它不是一个常量，也就是说无法和`constexpr`结合：

```cpp
thread_local int tv;
static int sv;

int main() {
	constexpr int* sp = &sv; // 编译成功，sv的地址在编译时确定
	constexpr int* tp = &tv; // 编译失败，tv的地址在运行时确定
}
```

最后来说明一下线程局部存储对象的初始化和销毁。在同一个线程中，一个线程局部存储对象只会初始化一次，即使在某个函数中被多次调用。这一点和单线程程序中的静态对象非常相似。相对应的，对象的销毁也只会发生一次，通常发生在线程退出的时刻。下面来看一个例子：

```cpp
#include <iostream>
#include <string>
#include <thread>
#include <mutex>

std::mutex g_out_lock;

struct RefCount {
	RefCount(const char* f) : i(0), func(f) {
		std::lock_guard<std::mutex> lock(g_out_lock);
		std::cout << std::this_thread::get_id() << "|" << func << " : ctor i(" << i << ")" << std::endl;
	}
	~RefCount() {
		std::lock_guard<std::mutex> lock(g_out_lock);
		std::cout << std::this_thread::get_id() << "|" << func << " : dtor i(" << i << ")" << std::endl;
	}
	void inc() {
		std::lock_guard<std::mutex> lock(g_out_lock);
		std::cout << std::this_thread::get_id() << "|" << func << " : ref count add 1 to i(" << i << ")" << std::endl;
		i++;
	}
	int i;
	std::string func;
};
RefCount *lp_ptr = nullptr;

void foo(const char* f) {
	std::string func(f);
	thread_local RefCount tv(func.append("#foo").c_str());
	tv.inc();
}

void bar(const char* f) {
	std::string func(f);
	thread_local RefCount tv(func.append("#bar").c_str());
	tv.inc();
}

void threadfunc1() {
	const char* func = "threadfunc1";
	foo(func);
	foo(func);
	foo(func);
}

void threadfunc2() {
	const char* func = "threadfunc2"; 
	foo(func);
	foo(func);
	foo(func);
}

void threadfunc3() {
	const char* func = "threadfunc3";
	foo(func);
	bar(func);
	bar(func);
}

int main() {
	std::thread t1(threadfunc1);
	std::thread t2(threadfunc2);
	std::thread t3(threadfunc3);
	
	t1.join();
	t2.join();
	t3.join();
}
```

下面是在Windows上的运行结果：

```txt
27300|threadfunc1#foo : ctor i(0)
27300|threadfunc1#foo : ref count add 1 to i(0)
27300|threadfunc1#foo : ref count add 1 to i(1)
27300|threadfunc1#foo : ref count add 1 to i(2)
25308|threadfunc3#foo : ctor i(0)
25308|threadfunc3#foo : ref count add 1 to i(0)
25308|threadfunc3#bar : ctor i(0)
25308|threadfunc3#bar : ref count add 1 to i(0)
25308|threadfunc3#bar : ref count add 1 to i(1)
10272|threadfunc2#foo : ctor i(0)
10272|threadfunc2#foo : ref count add 1 to i(0)
10272|threadfunc2#foo : ref count add 1 to i(1)
10272|threadfunc2#foo : ref count add 1 to i(2)
27300|threadfunc1#foo : dtor i(3)
25308|threadfunc3#bar : dtor i(2)
25308|threadfunc3#foo : dtor i(1)
10272|threadfunc2#foo : dtor i(3)
```

从结果可以看出，线程`threadfunc1`和`threadfunc2`分别只调用了一次构造和析构函数，而且引用计数的递增也不会互相干扰，也就是说两个线程中线程局部存储对象是独立存在的。对于线程`threadfunc3`，它进行了两次线程局部存储对象的构造和析构，这两次分别对应`foo`和`bar`函数里的线程局部存储对象`tv`。可以发现，虽然这两个对象具有相同的对象名，但是由于不在同一个函数中，因此也应该认为是相同线程中不同的线程局部存储对象，它们的引用计数的递增同样不会相互干扰。

# 扩展的`inline`说明符（C++17）

## 定义非常量静态成员变量的问题

在C++17标准之前，定义类的非常量静态成员变量是一件让人头痛的事情，因为变量的声明和定义必须分开进行，比如：

```cpp
#include <iostream>
#include <string>

class X {
public:
	static std::string text;
};

std::string X::text{ "hello" };

int main() {
	X::text += " world";
	std::cout << X::text << std::endl;
}
```

在这里`static std::string text`是静态成员变量的声明，`std::string X::text{ "hello" }`是静态成员变量的定义和初始化。为了保证代码能够顺利地编译，我们必须保证静态成员变量的定义有且只有一份，稍有不慎就会引发错误，比较常见的错误是为了方便将静态成员变量的定义放在头文件中：

```cpp
#ifndef X_H
#define X_H
class X {
public:
	static std::string text;
};

std::string X::text{ "hello" };
#endif
``` 

将上面的代码包含到多个CPP文件中会引发一个链接错误，因为`include`是单纯的宏替换，所以会存在多份`X::text`的定义导致链接失败。对于一些字面量类型，比如整型、浮点类型等，这种情况有所缓解，至少对于它们而言常量静态成员变量是可以一边声明一边定义的：

```cpp
#include <iostream>
#include <string>

class X {
public:
	static const int num { 5 };
};

int main() {
	std::cout << X:num << std::endl;
}
```

虽然常量性能让它们方便地声明和定义，但却丢失了修改变量的能力。对于`std::string`这种非字面量类型，这种方法是无能为力的。

## 使用`inline`说明符

为了解决上面这些问题，C++17标准中增强了`inline`说明符的能力，它允许我们内联定义静态变量，例如：

```cpp
#include <iostream>
#include <string>

class X {
public:
	inline static std::string text{"hello"};
};

int main() {
	X::text += " world";
	std::cout << X::text << std::endl;
}
```

上面的代码可以成功编译和运行，而且即使将类`X`的定义作为头文件包含在多个CPP中也不会有任何问题。在这种情况下，编译器会在类`X`的定义首次出现时对内联静态成员变量进行定义和初始化。

# 常量表达式

## 常量的不确定性

在C++11标准以前，我们没有一种方法能够有效地要求一个变量或者函数在编译阶段就计算出结果。由于无法确保在编译阶段得出结果，导致很多看起来合理的代码却引来编译错误。这些场景主要集中在需要编译阶段就确定的值语法中，比如`case`语句、数组长度、枚举成员的值以及非类型的模板参数。让我们先看一看这些场景的代码：

```cpp
const int index0 = 0;
#define index1 1

// case语句
switch (argc) {
case index0:
	std::cout << "index0" << std::endl;
	break;
case index1;
	std::cout << "index1" << std::endl;
	break;
default:
	std::cout << "none" << std::endl;
}

const int x_size = 5 + 8;
#define y_size 6 + 7
// 数组长度
char buffer[x_size][y_size] = { 0 };

// 枚举成员
enum {
	enum_index0 = index0;
	enum_index1 = index1;
};
std::tuple<int, char> tp = std::make_tuple(4, '3');
// 非类型的模板参数
int x1 = std::get<index0>(tp);
char x2 = std::get<index1>(tp);
```

让人遗憾的是上面这些方法并不可靠。首先，C++程序员应该尽量少使用宏，因为预处理器对于宏只是简单的字符替换，完全没有类型检查，而且宏使用不当出现的错误难以排查。其次，对`const`定义的常量可能是一个运行时常量，这种情况下是无法在case语句以及数组长度等语句中使用的。让我们稍微修改一下上面的代码：

```cpp
int get_index0 () {
	return 0;
}

int get_index1 () {
	return 1;
}

int get_x_size() {
	return 5 + 8;
}

int get_y_size() {
	return 6 + 7;
}

const int index0 = get_index0();
#define index1 get_index1()

switch(argc) {
case index0:
	std::cout << "index0" << std::endl;
	break;
case index1:
	std::cout << "index1" << std::endl;
default:
	std::cout << "none" << std::endl;
}

const int x_size = get_x_size();
#define y_size get_y_size()
char buffer[x_size][y_size] = { 0 };

enum {
	enum_index0 = index0,
	enum_index1 = index1,
};

std::tuple<int, char> tp = std::make_tuple(4, '3');
int x1 = std::get<index0>(tp);
char x2 = std::get<index1>(tp);
```

像上面这种尴尬的情况不仅可能出现在我们的代码中，实际上标准库中也有这样的情况，其中`<limits>`就是一个典型的例子。在C语言中存在头文件`<limits.h>`，在这个头文件中用宏定义了各种整型类型的最大值和最小值，比如：

```cpp
#define UCHAR_MAX 0xff // unsigned char类型的最大值
```

我们可以用这些宏代替数字，让代码有更好的可读性。这其中就包括要求编译阶段必须确定值的语句，例如定义一个数组：

```cpp
char buffer[UCHAR_MAX] = { 0 };
```

代码编译起来没有任何障碍。但是正如上文中提到的，C++程序员应该尽量避开宏。标准库为我们提供了一个`<limits>`，使用它同样能获得unsigned char类型的最大值：

```cpp 
std::numeric_limits<unsigned char>::max()
```

但是，如果想用它来声明数组的大小是无法编译成功的：

```cpp
char buffer[std::numeric_limits<unsigned char>::max()] = {0};
```

为了解决以上常量无法确定的问题，C++标准委员会决定在C++11标准中定义一个新的关键字`constexpr`，它能够有效地定义常量表达式，并且达到类型安全、可移植、方便库和嵌入式系统开发的目的。

## `constexpr`值

`constexpr`值即常量表达式值，是一个用`constexpr`说明符声明的变量或者数据成员，它要求该值必须在编译期计算。另外，常量表达式值必须被常量表达式初始化。定义常量表达式值的方法非常简单，例如：

```cpp
constexpr int x = 42;
char buffer[x] = { 0 };
```

以上代码定义了一个常量表达式值`x`，并将其初始化为`42`，然后用`x`作为数组长度定义了数组buffer。从这段代码来看，`constexpr`和`const`是没有区别的，我们将关键字替换为`const`同样能达到目的：

```cpp 
const int x = 42;
char buffer[x] = { 0 };  
```

从结果来看确实如此，在使用常量表达式初始化的情况下`constexpr`和`const`拥有相同的作用。但是`const`并没有确保编译期常量的特性，所以在下面的代码中，它们会有不同的表现：

```cpp
int x1 = 42;
const int x2 = x1;    // 定义和初始化成功
char buffer[x2] = { 0 };    // 编译失败，x2无法作为数组长度
```

在GCC中，这段代码可以编译成功，但是MSVC和CLang则会编译失败。如果把`const`替换为`constexpr`，会有不同的情况发生：

```cpp
int x1 = 42;
constexpr int x2 = x1;   // 编译失败，x2无法用x1初始化
char buffer[x2] = { 0 };
```

## `constexpr`函数

`constexpr`不仅能用来定义常量表达式值，还能定义一个常量表达式函数，即`constexpr`函数，常量表达式函数的返回值可以在编译阶段就计算出来。不过在定义常量表示函数的时候，我们会遇到更多的约束规则（在C++14和后续的标准中对这些规则有所放宽）。

- **函数必须返回一个值，所以它的返回值类型不能是`void`。**

- **函数体必须只有一条语句：`return expr`，其中`expr`必须也是一个常量表达式。如果函数有形参，则将形参替换到`expr`中后，`expr`仍然必须是一个常量表达式。**

- **函数使用之前必须有定义。**

- **函数必须用`constexpr`声明。**

让我们来看一看下面这个例子:

```cpp
constexpr int max_unsigned_char() {
	return 0xff;
}

constexpr int square(int x) {
	return x * x;
}

constexpr int abs(int x) {
	return x > 0 ? x : -x;
}

int main() {
	char buffer1[max_unsigned_char()] = { 0 };
	char buffer2[square(5)] = { 0 };
	char buffer3[abs(-8)] = { 0 };
}
```

由于标准规定函数体中只能有一个表达式`return expr`，因此是无法使用`if`语句的，幸运的是用条件表达式也能完成类似的效果。

```cpp
constexpr int next(int x) {
	return ++x;
}

int g() {
	return 42;
}

constexpr int f() {
	return g();
}

constexpr int max_unsigned_char2();
enum {
	max_uchar = max_unsigned_char2()
}

constexpr int abs2(int x) {
	if (x > 0) {
		return x;
	} else {
		return -x;
	}
} 

constexpr int sum(int x) {
	int result = 0;
	while (x > 0) {
		result += x--;
	}
	return result;
}
```

以上`constexpr`函数都会编译失败。我们可以使用递归来完成循环的操作，现在就来重写sum函数：

```cpp
constexpr int sum(int x) {
	return x > 0 ? x + sum(x - 1) : 0;
}
```

需要强调一点的是，虽然常量表达式函数的返回值可以在编译期计算出来，但是这个行为并不是确定的。例如，当带形参的常量表达式函数接受了一个非常量实参时，常量表达式函数可能会退化为普通函数：

```cpp
constexpr int square(int x) {
	return x * x;
}

int x = 5;
std::cout << square(x);
```

这里也存在着不确定性，因为GCC依然能在编译阶段计算`square`的结果，但是MSVC和CLang则不行。

有了常量表达式函数的支持，C++标准对STL也做了一些改进，比如在`<limits>`中增加了`constexpr`声明，正因如此下面的代码也可以顺利编译成功了：

```cpp
char buffer[std::numeric_limits<unsigned char>::max()] = { 0 };
```

## `constexpr`构造函数

`constexpr`可以声明基础类型从而获得常量表达式值，除此之外`constexpr`还能够声明用户自定义类型，例如：

```cpp
struct X {
	int x1;
};

constexpr X x = { 1 };
char buffer[x.x1] = { 0 };
```

以上代码自定义了一个结构体`X`，并且使用`constexpr`声明和初始化了变量`x`。到目前为止一切顺利，不过有时候我们并不希望成员变量被暴露出来，于是修改了`X`的结构：

```cpp
class X {
public:
	X() : x1(5) {}
	int get() const {
		return x1;
	}
private:
	int x1;
};

constexpr X x;  // 编译失败，X不是字面类型
char buffer[x.get()] = { 0 };  // 编译失败，x.get()无法在编译阶段计算
```

经过修改的代码不能通过编译了，因为`constexpr`说明符不能用来声明这样的自定义类型。解决上述问题的方法很简单，只需要用`constexpr`声明`X`类的构造函数，也就是声明一个常量表达式构造函数，当然这个构造函数也有一些规则需要遵循。

- **构造函数必须用`constexpr`声明。**

- **构造函数初始化列表中必须是常量表达式。**

- **构造函数的函数体必须为空（这一点基于构造函数没有返回值，所以不存在`return expr`）。**

根据以上规则让我们改写类X：

```cpp
class X {
public:
	constexpr X() : x1(5) {}
	constexpr X(int i) : x1(i) {}
	constexpr int get() const {
		return x1;
	}
private:
	int x1;
};

constexpr X x;
char buffer[x.get()] = { 0 };
```

在C++11中，`constexpr`会自动给函数带上`const`属性。请注意，常量表达式构造函数拥有和常量表达式函数相同的退化特性，当它的实参不是常量表达式的时候，构造函数可以退化为普通构造函数，当然，这么做的前提是类型的声明对象不能为常量表达式值：

```cpp 
int i = 8;
constexpr X x(i);   // 编译失败，不能使用constexpr声明
X y(i);    // 编译成功
```

由于i不是一个常量，因此X的常量表达式构造函数退化为普通构造函数，这时对象`x`不能用`constexpr`声明，否则编译失败。最后需要强调的是，使用`constexpr`声明自定义类型的变量，必须确保这个自定义类型的析构函数是平凡的，否则也是无法通过编译的。平凡析构函数必须满足下面3个条件。

- **自定义类型中不能有用户自定义的析构函数。**

- **析构函数不能是虚函数。**

- **基类和成员的析构函数必须都是平凡的。**

## 对浮点的支持

在constexpr说明符被引入之前，C++程序员经常使用`enum hack`来促使编译器在编译阶段计算常量表达式的值。但是因为`enum`只能操作整型，所以一直无法完成对于浮点类型的编译期计算。`constexpr`说明符则不同，它支持声明浮点类型的常量表达式值，而且标准还规定其精度必须至少和运行时的精度相同，例如：

```cpp
constexpr double sum(double x) {
	return x > 0 ? x + sum(x - 1) : 0;
}

constexpr double x = sum(5);
```

## C++14标准对常量表达式函数的增强

C++11标准对常量表达式函数的要求可以说是非常的严格，这一点影响该特性的实用性。幸好这个问题在C++14中得到了非常巨大的改善，C++14标准对常量表达式函数的改进如下。

- **函数体允许声明变量，除了没有初始化、`static`和`thread_local`变量。**

- **函数允许出现`if`和`switch`语句，不能使用`go`语句。**

- **函数允许所有的循环语句，包括`for`、`while`、`do-while`。**

- **函数可以修改生命周期和常量表达式相同的对象。**

- **函数的返回值可以声明为`void`。**

- **`constexpr`声明的成员函数不再具有`const`属性。**

因为这些改进的发布，在C++11中无法成功编译的常量表达式函数，在C++14中可以编译成功了：

```cpp
constexpr int abd(int x) {
	if (x > 0) {
		return x;
	} else {
		return -x;
	}
}

constexpr int sum(int x) {
	int result = 0;
	while (x > 0) {
		result += x--;
	}
	return result;
}

char buffer1[sum(5)] = { 0 };
char buffer2[abs(-5)] = { 0 };
```

以上代码中的`abs`和`sum`函数相比于前面使用条件表达式和递归方法实现的函数更加容易阅读和理解了。

```cpp
constexpr int next(int x) {
	return ++x;
}

char buffer[next(5)] = { 0 };
```

原来由于`++x`不是常量表达式，因此无法编译通过的问题也消失了，这就是基于第4点规则。需要强调的是，对于常量表达式函数的增强同样也会影响常量表达式构造函数：

```cpp
#include <iostream>

class X {
public:
	constexpr X() : x1(5) {}
	constexpr X(int i) : x1(0) {
		if (i > 0) {
			x1 = 5;
		} else {
			x1 = 8;
		}
	}
	constexpr void set(int i) {
		x1 = i;
	}
	constexpr int get() const {
		return x1;
	}
private:
	int x1;
};

constexpr X make_x() {
	X x;
	x.set(42);
	return x;	
}

int main() {
	constexpr X x1(-1);
	constexpr X x2 = make_x();
	constexpr int a1 = x1.get();
	constexpr int a2 = x2.get();
	std::cout << a1 << std::endl;
	std::cout << a2 << std::endl;
}
```

请注意，`main`函数里的4个变量`x1`、`x2`、`a1`和`a2`都有`constexpr`声明，也就是说它们都是编译期必须确定的值。有了这个前提条件，我们再来分析这段代码的神奇之处。首先对于常量表达式构造函数，我们发现可以在其函数体内使用if语句并且对`x1`进行赋值操作了。可以看到返回类型为`void`的`set`函数也被声明为`constexpr`了，这也意味着该函数能够运用在`constexpr`声明的函数体内，`make_x`函数就是利用了这个特性。根据规则4和规则6，`set`函数也能成功地修改`x1`的值了。让我们来看一看GCC生成的中间代码：

```cpp
main() {
	int D.39319;
	
	{
		const struct X x1;
		const struct X x2;
		const int a1;
		const int a2;
		
		try {
			x1.x1 = 8;
			x2.x1 = 42;
			a1 = 8;
			a2 = 42;
			_1 = std::basic_ostream<char>::operator<< (&cout, 8);
			std::basic_ostream<char>::operator<< (_1, endl);
			_2 = std::basic_ostream<char>::operator<< (&cout, 42);
			std::basic_ostream<char>::operator<< (_2, endl);
		} finally {
			x1 = { CLOBBER };
			x2 = { CLOBBER };
		}
	}
	D.39319 = 0;
	return D.39319;
}
```

从上面的中间代码可以清楚地看到，编译器直接给`x1.x1`、`x2.x1`、`a1`、`a2`进行了赋值，并没有运行时的计算操作。

## `constexpr` `lambdas`表达式

从C++17开始，`lambda`表达式在条件允许的情况下都会隐式声明为`constexpr`。这里所说的条件，即是上一节中提到的常量表达式函数的规则，本节里就不再重复论述。结合`lambda`的这个新特性，先看一个简单的例子：

```cpp
constexpr int foo() {
	return []() { return 58; }();
}

auto get_size = [](int i) { return i * 2; };
char buffer1[foo()] = { 0 };
char buffer2[get_size(5)] = { 0 };
```

可以看到，以上代码定义的是一个“普通”的`lambda`表达式，但是在C++17标准中，这些“普通”的`lambda`表达式却可以用在常量表达式函数和数组长度中，可见该`lambda`表达式的结果在编译阶段已经计算出来了。实际上这里的`[](int i) { return i * 2; }`相当于：

```cpp
class GetSize {
public:
	constexpr int operator() (int i) const {
		return i * 2;
	}
};
```

当`lambda`表达式不满足`constexpr`的条件时，`lambda`表达式也不会出现编译错误，它会作为运行时`lambda`表达式存在：

```cpp
// 情况1
int i = 5;
auto get_size = [](int i) { return i * 2; };
char buffer1[get_size(i)] = { 0 }; // 编译失败，get_size需要运行时调用
int a1 = get_size(i);

// 情况2
auto get_count = []() {
	static int x = 5;
	return x;
};
int a2 = get_count();
```

值得注意的是，我们也可以强制要求`lambda`表达式是一个常量表达式，用`constexpr`去声明它即可。这样做的好处是可以检查`lambda`表达式是否有可能是一个常量表达式，如果不能则会编译报错，例如：

```cpp
auto get_size = [](int i) constexpr -> int { return i * 2; };
char buffer2[get_size(5)] = { 0 };

auto get_count = []() constexpr -> int {
	static int x = 5;    // 编译失败，x是一个static变量
	return x;
};
int a2 = get_count();
```

## `constexpr`的内联属性

在C++17标准中，`constexpr`声明静态成员变量时，也被赋予了该变量的内联属性，例如：

```cpp
class X {
public:
	static constexpr int num{ 5 };
};
```

以上代码在C++17中等同于：

```cpp
class X {
public:
	inline static constexpr int num{ 5 };
};
```

那么问题来了，自C++11标准推行以来`static constexpr intnum{ 5 }`这种用法就一直存在了，那么同样的代码在C++11和C++17中究竟又有什么区别呢？

```cpp
class X {
public:
	static constexpr int num{ 5 };
};
```

代码中，`num`是只有声明没有定义的，虽然我们可以通过`std::cout << X::num << std::endl`输出其结果，但这实际上是编译器的一个小把戏，它将`X::num`直接替换为了5。如果将输出语句修改为`std::cout << &X::num << std::endl`，那么链接器会明确报告`X::num`缺少定义。但是从C++17开始情况发生了变化，`static constexpr int num{5}`既是声明也是定义，所以在C++17标准中`std::cout << &X::num << std::endl`可以顺利编译链接，并且输出正确的结果。值得注意的是，对于编译器而言为`X::num`产生定义并不是必需的，如果代码只是引用了`X::num`的值，那么编译器完全可以使用直接替换为值的技巧。只有当代码中引用到变量指针的时候，编译器才会为其生成定义。

## `if constexpr`

`if constexpr`是C++17标准提出的一个非常有用的特性，可以用于编写紧凑的模板代码，让代码能够根据编译时的条件进行实例化。这里有两点需要特别注意。

- `if constexpr`的条件必须是编译期能确定结果的常量表达式。

- 条件结果一旦确定，编译器将只编译符合条件的代码块。

由此可见，该特性只有在使用模板的时候才具有实际意义，若是用在普通函数上，效果会非常尴尬，比如：

```cpp
void check1(int i) {
	if constexpr (i > 0) {    // 编译失败，不是常量表达式
		std::cout << "i > 0" << std::endl;
	} else {
		std::cout << "i <= 0" << std::endl;
	}
}

void check2() {
	if constexpr (sizeof(int) > sizeof(char)) {
		std::cout << "sizeof(int) > sizeof(char)" << std::endl;
	} else {
		std::cout << "sizeof(int) <= sizeof(char)" << std::endl;
	}
}
```

对于函数`check1`，由于`if constexpr`的条件不是一个常量表达式，因此无法编译通过。而对于函数`check2`，这里的代码最后会被编译器省略为：

```cpp
void check2() {
	std::cout << "sizeof(int) > sizeof(char)" << std::endl;
}
```

但是当`if constexpr`运用于模板时，情况将非常不同。来看下面的例子：

```cpp
#include <iostream>

template<class T> bool is_same_value(T a, T b) {
	return a == b;
}

template<> bool is_same_value<double>(double a, double b) {
	if (std::abs(a - b) < 0.0001) {
		return true;
	} else {
		return false;
	}
}

int main() {
	double x = 0.1 + 0.1 + 0.1 - 0.3;
	std::cout << std::boolalpha;
	std::cout << "is_same_value(5, 5) : " << is_same_value(5, 5) << std::endl;
	std::cout << "x == 0.0      : " << (x == 0.) << std::endl;
	std::cout << "is_same_value(x, 0.) : " << is_same_value(x, 0.) << std::endl;
}
```

计算结果如下：

```txt
is_same_value(5, 5) : true
x == 0.0 : false
is_same_value(x, 0.) : true
```


浮点数的比较和整数是不同的，通常情况下它们的差小于某个阈值就认为两个浮点数相等。我们把`is_same_value`写成函数模板，并且对`double`类型进行特化。这里如果使用`if constexpr`表达式，代码会简化很多而且更加容易理解，让我们看一看简化后的代码：

```cpp
#include <type_traits>
template<class T> bool is_same_value(T a, T b) {
	if constexpr (std::is_same<T, double>::value) {
		if (std::abs(a - b) < 0.0001) {
			return true;
		} else {
			return false;
		}
	} else {
		return a == b;
	}
}
```

在上面这段代码中，直接使用`if constexpr`判断模板参数是否为`double`，如果条件成立，则使用`double`的比较方式；否则使用普通的比较方式，代码变得简单明了。再次强调，这里的选择是编译期做出的，一旦确定了条件，那么就只有被选择的代码块才会被编译；另外的代码块则会被忽略。说到这里，需要提醒读者注意这样一种陷阱：

```cpp
#include <iostream>
#include <type_traits>
template<class T> auto minus(T a, T b) {
	if constexpr (std::is_same<T, double>::value) {
		if (std::abs(a - b) < 0.0001) {
			return 0.;
		} else {
			return a - b;	
		}
	} else {
		return static_cast<int>(a - b);
	}
}

int main() {
	std::cout << minus(5.6, 5.11) << std::endl;
	std::cout << minus(5.60002, 5.600011) << std::endl;
	std::cout << minus(6,5) << std::endl;
}
```

以上是一个带精度限制的减法函数，当参数类型为`double`且计算结果小于0.0001的时候，我们就可以认为计算结果为0。当参数类型为整型时，则不用对精度做任何限制。上面的代码编译运行没有任何问题，因为编译器根据不同的类型选择不同的分支进行编译。但是如果修改一下上面的代码，结果可能就很难预料了：

```cpp
template<class T> auto minus(T a, T b) {
	if constexpr (std::is_same<T, double>::value) {
		if (std::abs(a - b) < 0.0001) {
			return 0.;
		} else {
			return a - b;
		}
	}
	return static_cast<int>(a - b);
}
```


上面的代码删除了`else`关键词而直接将`else`代码块提取出来，不过根据以往运行时if的经验，它并不会影响代码运行的逻辑。遗憾的是，这种写法有可能导致编译失败，因为它可能会导致函数有多个不同的返回类型。当实参为整型时一切正常，编译器会忽略if的代码块，直接编译`return static_cast<int>(a − b)`，这样返回类型只有int一种。但是当实参类型为double的时候，情况发生了变化。if的代码块会被正常地编译，代码块内部的返回结果类型为double，而代码块外部的return `static_cast<int>(a − b)`同样会照常编译，这次的返回类型为int。编译器遇到了两个不同的返回类型，只能报错。

和运行时if的另一个不同点：`if constexpr`不支持短路规则。这在程序编写时往往也能成为一个陷阱:

```cpp
#include <iostream>
#include <string>
#include <type_traits>

template<class T>auto any2i(T t) {
	if constexpr (std::is_same<T, std::string>::value && T::npos == -1) {
		return atoi(t.c_str());
	} else {
		return t;
	}
}

int main() {
	std::cout << any2i(std::string("6")) << std::endl;
	std::cout << any2i(6) << std::endl;
}
```

上面的代码很好理解，函数模板`any2i`的实参如果是一个`std::string`，那么它肯定满足`std::is_same<T,std::string>::value && T::npos == −1`的条件，所以编译器会编译if分支的代码。如果实参类型是一个`int`，那么`std::is_same<T, std::string>::value`会返回false，根据短路规则，if代码块不会被编译，而是编译`else`代码块的内容。一切看起来是那么简单直接，但是编译过后会发现，代码`std::cout<< any2i(std:: string("6")) << std::endl`顺利地编译成功，`std::cout << any2i(6) << std::endl`则会编译失败，因为if constexpr不支持短路规则。当函数实参为int时，`std::is_same<T, std::string>::value`和`T::npos == −1`都会被编译，由于`int::npos`显然是一个非法的表达式，因此会造成编译失败。这里正确的写法是通过嵌套`if constexpr`来替换上面的操作：

```cpp
template<class T> auto any2i(T t) {
	if constexpr (std::is_same<T, std::string>::value) {
		return atoi(t.c_str());
	} else {
		return t;
	}
}
```

## 允许constexpr虚函数

在C++20标准之前，虚函数是不允许声明为`constexpr`的。看似有道理的规则其实并不合理，因为虚函数很多时候可能是无状态的，这种情况下它是有条件作为常量表达式被优化的，比如下面这个函数：

```cpp
struct X {
	virtual int f() const { return 1; }
};

int main() {
	X x;
	int i = x.f();
}
```

上面的代码会先执行`X::f`函数，然后将结果赋值给i，它的GIMPLE中间的代码如下：

```cpp
main() {
	int D.2137;
	
	{
		struct X x;
		int i;
		try {
			_1 = &_ZTV1X + 16;
			x._vptr.X = _1;
			i = X::f(&x); // 注意此处赋值
		} finally {
			x = {CLOBBER};
		}
	}
	D.2137 = 0;
	return D.2137;
}

X::f (const struct X* const this) {
	int D.2139;
	D.2139 = 1;
	
	return D.2139;
}
```

观察上面的两份代码，虽然`X::f`是一个虚函数，但是它非常适合作为常量表达式进行优化。这样一来，`int i = x.f();`可以被优化为`int i = 1;`，减少一次函数的调用过程。可惜在C++17标准中不允许我们这么做，直到C++20标准明确允许在常量表达式中使用虚函数，所以上面的代码可以修改为：

```cpp
struct X {
	constexpr virtual int f() const { return 1; }
};

int main() {
	constexpr X x;
	int i = x.f();
}
```

它的中间代码也会优化为：

```cpp
main() {
	int D.2138;
	
	{
		const struct X x;
		int i;
		try {
			_1 = &_ZTV1X + 16;
			x._vptr.X = _1;
			i = 1; // 注意此处赋值
		} finally {
			x = {CLOBBER};
		}
	}
	D.2138 = 0;
	return D.2138;
}
```

从中间代码中可以看到，i被直接赋值为1，在此之前并没有调用`X::f`函数。另外值得一提的是，`constexpr`的虚函数在继承重写上并没有其他特殊的要求，`constexpr`的虚函数可以覆盖重写普通虚函数，普通虚函数也可以覆盖重写`constexpr`的虚函数，例如：

```cpp
struct X1 {
	virtual int f() const = 0;
};

struct X2 : public X1 {
	constexpr virtual int f() const { return 2; }
};

struct X3 : public X2 {
	virtual int f() const { return 3; }
};

struct X4 : public X3 {
	constexpr virtual int f() const { return 4; }
};

constexpr int (X1::*pf)() const &X1::f;

constexpr X2 x2;
static_assert(x2.f() == 2);
static_assert((x2.*pf)() == 2);

constexpr X1 const& r2 = x2;
static_assert(r2.f() == 2);
static_assert((r2.*pf)() == 2);

constexpr X1 const* p2 = &x2;
static_assert(p2->f() == 2);
static_assert((p2->*pf)() == 2);

constexpr X4 x4;
static_assert(x4.f() == 4);
static_assert((x4.*pf)() == 4);

constexpr X1 const& r4 = x4;
static_assert(r4.f() == 4);
static_assert((r4.*pf)() == 4);

constexpr X1 const* p4 = &x4;
static_assert(p4->f() == 4);
static_assert((p4->*pf)() == 4);
```

最后要说明的是，GCC无论在C++17还是C++20标准中都可以顺利编译通过，而CLang在C++17中会给出`constexpr`无法用于虚函数的错误提示。

## 允许在`constexpr`函数中出现`Try-catch`

在C++20标准以前`Try-catch`是不能出现在`constexpr`函数中的，例如：

```cpp
constexpr int f(int x) {
	try { return x + 1; }
	catch(...) { return 0; }
}
```

不过似乎编译器对此规则的态度都十分友好，当我们用C++17标准去编译这份代码时，编译器会编译成功并给出一个友好的警告，说明这条特性需要使用C++20标准。C++20标准允许`Try-catch`存在于`constexpr`函数，但是`throw`语句依旧是被禁止的，所以`try`语句是不能抛出异常的，这也就意味着`catch`永远不会执行。实际上，当函数被评估为常量表达式的时候`Try-catch`是没有任何作用的。

## 允许在`constexpr`中进行平凡的默认初始化

从C++20开始，标准允许在`constexpr`中进行平凡的默认初始化，这样进一步减少`constexpr`的特殊性。例如：

```cpp
struct X {
	bool val;
};

void f() {
	X x;
}

f();
```

上面的代码非常简单，在任何环境下都可以顺利编译。不过如果将函数f改为：

```cpp 
constexpr void f() {
	X x;
}
```

那么在C++17标准的编译环境就会报错，提示x没有初始化，它需要用户提供一个构造函数。当然这个问题在C++17标准中也很容易解决，例如修改X为：

```cpp
struct X {
	bool val = false;
};
```

值得一提的是，虽然标准放松了对`constexpr`上下文对象默认初始化的要求，但是我们依然应该养成声明对象时随手初始化的习惯，避免让代码出现未定义的行为。

## 允许在`constexpr`中更改联合类型的有效成员

在C++20标准之前对`constexpr`的另外一个限制就是禁止更改联合类型的有效成员，例如：

```cpp
union Foo {
	int i;
	float f;
};

constexpr int use() {
	Foo foo{};
	foo.i = 3;
	foo.f = 1.2f;   // C++20之前编译失败
	return 1;
}
```

在上面的代码中，`foo`是一个联合类型对象，`foo.i = 3;`首次确定了有效成员为`i`，这没有问题，接下来代码`foo.f = 1.2f;`改变有效成员为`f`，这就违反了标准中关于不能更改联合类型的有效成员的规则，所以导致编译失败。现在C++20标准已经删除了这条规则，以上代码可以编译成功。实际编译过程中，只有CLang会在C++17标准中对以上代码报错，而GCC和MSVC均能用C++17和C++20标准编译成功。

C++20标准对`constexpr`做了很多修改，除了上面提到的修改以外，还修改了一些并不常用的地方，包括允许`dynamic_cast`和`typeid`出现在常量表达式中；允许在`constexpr`函数使用未经评估的内联汇编。



## 使用`consteval`声明立即函数

前面我们曾提到过，`constexpr`声明函数时并不依赖常量表达式上下文环境，在非常量表达式的环境中，函数可以表现为普通函数。不过有时候，我们希望确保函数在编译期就执行计算，对于无法在编译期执行计算的情况则让编译器直接报错。于是在C++20标准中出现了一个新的概念——立即函数，该函数需要使用`consteval`说明符来声明：

```cpp
consteval int sqr(int n) {
	return n * n;
}

constexpr int r = sqr(100); // 编译成功
int x = 100;
int r2 = sqr(x);    // 编译失败
```

在上面的代码中`sqr(100);`是一个常量表达式上下文环境，可以编译成功。相反，因为`sqr(x);`中的`x`是可变量，不能作为常量表达式，所以编译器抛出错误。要让代码成功编译，只需要给`x`加上`const`即可。需要注意的是，如果一个立即函数在另外一个立即函数中被调用，则函数定义时的上下文环境不必是一个常量表达式，例如：

```cpp 
consteval int sqrsqr(int n) {
	return sqr(sqr(n));
}
```

`sqrsqr`是否能编译成功取决于如何调用，如果调用时处于一个常量表达式环境，那么就能通过编译：

```cpp
int y = sqrsqr(100);
```

反之则编译失败：

```cpp
int y = sqrsqr(x);
```

lambda表达式也可以使用consteval说明符：

```cpp
auto sqr = [](int n) consteval { return n * n; };
int r = sqr(100);
auto f = sqr; // 编译失败，尝试获取立即函数的函数地址
```

## 使用`constinit`检查常量初始化

在C++中有一种典型的错误叫作“Static Initialization Order Fiasco”，指的是因为静态初始化顺序错误导致的问题。因为这种错误往往发生在main函数之前，所以比较难以排查。举一个典型的例子，假设有两个静态对象x和y分别存在于两个不同的源文件中。其中一个对象x的构造函数依赖于对象y。没错，就是这样，现在我们有50%的可能性会出错，因为我们没有办法控制哪个对象先构造。如果对象x在y之前构造，那么就会引发一个未定义的结果。为了避免这种问题的发生，我们通常希望使用常量初始化程序去初始化静态变量。不幸的是，常量初始化的规则很复杂，需要一种方法帮助我们完成检查工作，当不符合常量初始化程序的时候可以在编译阶段报错。于是在C++20标准中引入了新的`constinit`说明符。

正如上文所描述的`constinit`说明符主要用于具有静态存储持续时间的变量声明上，它要求变量具有常量初始化程序。首先，`constinit`说明符作用的对象是必须具有静态存储持续时间的，比如：

```cpp
constinit int x = 11; // 编译成功，全局变量具有静态存储持续
int main() {
	constinit static int y = 42;  // 编译成功，静态变量具有静态存储持续
	constinit int z = 7;  // 编译失败，局部变量是动态分配的
}
```

其次，`constinit`要求变量具有常量初始化程序：

```cpp
const char* f() { reutrn "hello"; }
constexpr const char* g() { return "cpp"; }
constinit const char* str1 = f(); // 编译错误，f()不是一个常量初始化程序
constinit const char* str2 = g(); // 编译成功
```

`constinit`还能用于非初始化声明，以告知编译器`thread_local`变量已被初始化：

```cpp
extern thread_local constinit int x;
int f() { return x; }
```

最后值得一提的是，虽然`constinit`说明符一直在强调常量初始化，但是初始化的对象并不要求具有常量属性。

## 判断常量求值环境

`std::is_constant_evaluated`是C++20新加入标准库的函数，它用于检查当前表达式是否是一个常量求值环境，如果在一个明显常量求值的表达式中，则返回true；否则返回false。该函数包含在`<type_traits>`头文件中，虽然看上去像是一个标准库实现的函数，但实际上调用的是编译器内置函数：

```cpp
constexpr inline bool is_constant_evaluated() noexcept {
	return __builtin_is_constant_evaluated();
}
```

该函数通常会用于代码优化中，比如在确定为常量求值的环境时，使用`constexpr`能够接受的算法，让数值在编译阶段就得出结果。而对于其他环境则采用运行时计算结果的方法。提案文档中提供了一个很好的例子：

```cpp
#include <cmath>
#include <type_traits>
constexpr double power(double b, int x) {
	if (std::is_constant_evaluated() && x >= 0) {
		double r = 1.0, p = b;
		unsigned u = (unsigned) x;
		while (u != 0) {
			if (u & 1) r *= p;
			u /= 2;
			p *= p;
		}
		return r;
	} else {
		return std::pow(b, (double) x);
	}
}

int main() {
	constexpr double kilo = power(10.0, 3) // 常量求值
	int n = 3;
	double mucho = power(10.0, n); // 非常量求值
	return 0;
}
```

在上面的代码中，`power`函数根据`std::is_constant_evaluated()`和`x >= 0`的结果选择不同的实现方式。其中，`kilo = power(10.0, 3);`是一个常量求值，所以`std::is_ constant_evaluated() && x >= 0`返回`true`，编译器在编译阶段求出结果。反之，`mucho = power(10.0, n)`则需要调用`std::pow`在运行时求值。让我们通过中间代码看一看编译器具体做了什么：

```cpp
main() {
	int D.25691;
	{
		const double kilo;
		int n;
		double mucho;
		kilo = 1.0e+3;  // 直接赋值
		n = 3;
		mucho = power(1.0e+1, n);  // 运行时计算
		D.25691 = 0;
		return D.25691;
	}
	D.25691 = 0;
	return D.25691;
}

power(double b, int x) {
	bool retval.0;
	bool iftmp.1;
	double D.25706;
	{
		_1 = std::is_constant_evaluated();
		if (_1 != 0) goto <D.25697>; else goto <D.25695>;
		<D.25697>:
		if (x >= 0) goto <D.25698>; else goto <D.25695>;
		<D.25698>:
		iftmp.1 = 1;
		goto <D.25696>;
		<D.25695>:
		iftmp.1 = 0;
		<D.25696>:
		retval.0 = iftmp.1;
		if (retval.0 != 0) goto <D.25699>; else goto <D.25700>;
		<D.25699>:
		{
			// … 这里省略power函数的相关算法，虽然算法生成代码了，但是并没有调用到
			return D.25706;
		}
		<D.25700>:
		_3 = (double) x; 
		D.25706 = pow(b, _3);
		return D.25706;
	}
}

std::is_constant_evaluated() {
	bool D.25708;
	try {
		D.25708 = 0;
		return D.25708;
	} catch {
		<<<eh_must_not_throw (terminate) >>>
	}
}
```

观察上面的中间代码，首先让我们注意到的就是main函数中kilo和mucho赋值形式的不同。正如我们刚才讨论的那样，对于kilo的结果编译器在编译期已经计算完成，所以这里是直接为1.0e+3，而对于mucho则需要调用`std::power`函数。接着，我们可以观察`std::is_constant_evaluated()`这个函数的实现，很明显编译器让它直接返回0（也就是false），在代码中实现的power函数虽然有`std::is_constant_ evaluated()`结果为true时的算法实现，但是却永远不会被调用。因为当`std::is_constant_evaluated()`为true时，编译器计算了函数结果；反之函数会交给`std::power`计算结果。

在了解了`std::is_constant_evaluated()`的用途之后，我们还需要弄清楚何为明显常量求值。只有弄清楚这个概念，才可能合理运用`std::is_constant_ evaluated()`函数。明显常量求值在标准文档中列举了下面几个类别。

- **常量表达式，这个类别包括很多种情况，比如数组长度、`case`表达式、非类型模板实参等。**

- **`if constexpr`语句中的条件。**

- **`constexpr`变量的初始化程序。**

- **立即函数调用。**

- **约束概念表达式。**

- **可在常量表达式中使用或具有常量初始化的变量初始化程序。**

下面我们通过几个标准文档中的例子来体会以上规则：

```cpp
template<bool> struct X {};
X<std::is_constant_evaluated()> x; // 非类型模板实参，函数返回true，最终类型为 X<true>
int y;
constexpr int f() {
	const int n = std::is_constant_evaluated() ? 13 : 17; // n是13
	int m = std::is_constant_evaluated() ? 13 : 17; // m可能是13或者17，取决于函数环境
	char arr[n] = {};   // char[13]
	return m + sizeof(arr);
}

int p = f();    // m是13；p结果如下26
int q = p + f();  // m是17；q结果如下56
```



最后需要注意的是，如果当判断是否为明显常量求值时存在多个条件，那么编译器会试探`std::is_constant_evaluated()`两种情况求值，比如：

```cpp 
int y;
const int a = std::is_constant_evaluated() ? y : 1; // 函数返回false，a运行时初始化为1
const int b = std::is_constant_evaluated() ? 2 : y; // 函数返回true，b编译时初始化为2
```

当对a求值时，编译器试探`std::is_constant_evaluated()== true`的情况，发现y会改变a的值，所以最后选择`std::is_constant_evaluated() == false；`当对b求值时，编译器同样试探`std::is_constant_evaluated() == true`的情况，发现b的结果恒定为2，于是直接在编译时完成初始化。


# 确定的表达式求值顺序 （C++17）

## 表达式求值顺序的不确定性

在C++语言之父本贾尼·斯特劳斯特卢普的作品《C++程序设计语言（第4版）》中有一段这样的代码：

```cpp
void f2() {
	std::string s = "but I have heard it works even if you don't believe in it";
	s.replace(0, 4, "").replace(s.find("even"), 4, "only").replace(s.find(" don't"), 6, "");
	assert(s == "I have heard it works only if you believe in it");
	// OK
}
```

这段代码的本意是描述`std::string`成员函数`replace`的用法，但令人意想不到的是，在C++17之前它隐含着一个很大的问题，该问题的根源是表达式求值顺序。具体来说，是指一个表达式中的子表达式的求值顺序，而这个顺序在C++17之前是没有具体说明的，所以编译器可以以任何顺序对子表达式进行求值。比如说`foo(a, b, c)`，这里的`foo`、`a`、`b`和`c`的求值顺序是没有确定的。回到上面的替换函数，如果这里的执行顺序为：

```txt
1. replace(0, 4, "")
2. tmp1 = find("even")
3. replace(tmp1, 4, "only")
4. tmp2 = find(" don't")
5. replace(tmp2, 6, "")
```

那结果肯定是“I have heard it works only if you believe in it”，没有任何问题。但是由于没有对表达式求值顺序的严格规定，因此其求值顺序可能会变成：

```cpp
1. tmp1 = find("even")
2. tmp2 = find(" don't")
3. replace(0, 4, "")
4. replace(tmp1, 4, "only")
5. replace(tmp2, 6, "")
```

相应的结果就不是那么正确了，我们会得到“I have heard it works evenonlyyou donieve in it”。

虽然我们认为上面的表达式应该按照f()、g()、h()顺序对表达式求值，但是编译器对此并不买单，在它看来这个顺序可以是任意的。

## 表达式求值顺序详解

从C++17开始，函数表达式一定会在函数的参数之前求值。也就是说在`foo(a, b, c)`中，`foo`一定会在`a`、`b`和`c`之前求值。但是请注意，参数之间的求值顺序依然没有确定，也就是说`a`、`b`和`c`谁先求值还是没有规定。对于这一点我和读者应该是同样的吃惊，因为从提案文档上看来，有充分的理由说明从左往右进行参数列表的表达式求值的可行性。我想一个可能的原因是求值顺序的改变影响到代码的优化路径，比如内联决策和寄存器分配方式，对于编译器实现来说也是不小的挑战吧。不过既然标准已经这么定下来了，我们就应该去适应标准。**在函数的参数列表中，尽可能少地修改共享的对象，否则会很难确认实参的真实值。**

对于后缀表达式和移位操作符而言，表达式求值总是从左往右，比如：

```txt
E1[E2]
E1.E2
E1.*E2
E1->*E2
E1<<E2
E1>>E2
```

在上面的表达式中，子表达式求值`E1`总是优先于`E2`。而对于赋值表达式，这个顺序又正好相反，它的表达式求值总是从右往左，比如：

```txt
E1=E2
E1+=E2
E1-=E2
E1*=E2
E1/=E2
```

在上面的表达式中，子表达式求值E2总是优先于E1。这里虽然只列出了几种赋值表达式的形式，但实际上对于`E1@=E2`这种形式的表达式（其中`@`可以为`+`、`−`、`*`、`/`、`%`等）E2早于E1求值总是成立的。

对于`new`表达式，C++17也做了规定。对于：

```txt
new T(E)
```

这里`new`表达式的内存分配总是优先于T构造函数中参数E的求值。最后C++17还明确了一条规则：涉及重载运算符的表达式的求值顺序应由与之相应的内置运算符的求值顺序确定，而不是函数调用的顺序规则。

# 字面量优化（C++11～C++17）

## 十六进制浮点字面量

从C++11开始，标准库中引入了`std::hexfloat`和 `std::defaultfloat`来修改浮点输入和输出的默认格式化，其中 `std::hexfloat`可以将浮点数格式化为十六进制的字符串，而 `std::defaultfloat`可以将格式还原到十进制，以输出为例：

```cpp
#include <iostream>

int main() {
	double float_array[]{5.875, 1000, 0.117};
	for (auto elem : float_array) {
		std::cout << std::hexfloat << elem << " = " << std::defaultfloat << elem << std::endl;
	}
}
```

上面的代码分别使用`std::hexfloat`和`std::defaultfloat`格式化输出了数组x里的元素，输出结果如下

```txt
0x1.780000p+2 = 5.875
0x1.f40000p+9 = 1000
0x1.df3b64p-4 = 0.117
```

虽然C++11已经具备了在输入输出的时候将浮点数格式化为十六进制的能力，但遗憾的是我们并不能在源代码中使用十六进制浮点字面量来表示一个浮点数。幸运的是，这个问题在C++17标准中得到了解决：

```cpp
#include <iostream>

int main() {
	double float_array[] { 0x1.7p+2, 0x1.f4p+9, 0x1.df3b64p-4 };
	for (auto elem : float_array) {
		std::cout << std::hexfloat << elem 
			<< " = " << std::defaultfloat << elem << std::ednl;
	}
}
```

使用十六进制浮点字面量的优势显而易见，它可以更加精准地表示浮点数。例如，IEEE-754标准最小的单精度值很容易写为`0x1.0p−126`。当然了，十六进制浮点字面量的劣势也很明显，它不便于代码的阅读理解。总之，我们在C++17中可以根据实际需求选择浮点数的表示方法，当需要精确表示某个浮点数的时候可以采用十六进制浮点字面量，其他情况使用十进制浮点字面量即可。

## 二进制整数字面量

在C++14标准中定义了二进制整数字面量，正如十六进制（0x，0X）和八进制（0）都有固定前缀一样，二进制整数字面量也有前缀0b和0B。实际上GCC的扩展早已支持了二进制整数字面量，只不过到了C++14才作为标准引入：

```cpp
auto x = 0b11001101L + 0xcdl + 077LL + 42;
std::cout << "x = " << x << ", sizeof(x) = " << sizeof(x) << std::endl;
```

## 单引号作为整数

除了添加二进制整数字面量以外，C++14标准还增加了一个用单引号作为整数分隔符的特性，目的是让比较长的整数阅读起来更加容易。单引号整数分隔符对于十进制、八进制、十六进制、二进制整数都是有效的，比如：

```cpp
constexpr int x = 123'456;
static_assert(x == 0x1e'240);
static_assert(x == 036'11'00);
static_assert(x == 0b11'110'001'001'000'000);
```

值得注意的是，由于单引号在过去有用于界定字符的功能，因此这种改变可能会引起一些代码的兼容性问题，比如：

```cpp
#include <iostream>

#define M(x, ...) __VA_ARGS__
int x[2] = { M(1'2, 3'4) };

int main() {
	std::cout << "x[0] = " << x[0] << ", x[1] = " << x[1] << std::endl;
}
```

上面的代码在C++11和C++14标准下编译运行的结果不同，在C++11标准下输出结果为`x[0] = 0, x[1] = 0`，而在C++14标准下输出结果为`x[0] = 34, x[1] = 0`。这个现象很容易解释，在C++11中`1'2,3'4`是一个参数，所以`__VA_ARGS__`为空，而在C++14中它是两个参数12和34，所以`__VA_ARGS__`为34。虽然会引起一点兼容性问题，但是读者不必过于担心，上面这种代码很少会出现在真实的项目中，大部分情况下我们还是可以放心地将编程环境升级到C++14或者更高标准的，只不过如果真的出现了编译错误，不妨留意一下是不是这个问题造成的。

## 原生字符串字面量

过去想在C++中嵌入一段带格式和特殊符号的字符串是一件非常令人头痛的事情，比如在程序中嵌入一份HTML代码，我们不得不写成这样：

```cpp
char hello_world_html[] = 
"<!DOCTYPE html>\r\n"
"<html lang = \"en\">\r\n"
" <head>\r\n"
" <meta charset = \"utf-8\">\r\n"
" <meta name = \"viewport\" content = \"width=device-width, initial-scale=1, user-scalable=yes\">\r\n"
" <title>Hello World!</title>\r\n"
" </head>\r\n"
" <body>\r\n"
" Hello World!\r\n"
" </body>\r\n"
"</html>\r\n";
```

原生字符串字面量并不是一个新的概念，比如在Python中已经支持在字符串之前加R来声明原生字符串字面量了。使用原生字符串字面量的代码会在编译的时候被编译器直接使用，也就是说保留了字符串里的格式和特殊字符，同时它也会忽略转移字符，概括起来就是所见即所得。

声明原生字符串字面量的语法很简单，即`prefix R"delimiter(raw_ characters)delimiter"`，这其中`prefix`和`delimiter`是可选部分，我们可以忽略它们，所以最简单的原生字符串字面量声明是`R"(raw_characters)"`。以上面的HTML字符串为例：

```cpp
char hello_world_html[] = R"(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=yes">
<title>Hello World!</title>
</head>
<body>
Hello World!
</body>
</html>
)";
```

从上面的代码可以看到，原生字符串中不需要\r\n，也不需要对引号使用转义字符，编译后字符串的内容和格式与代码里的一模一样。读者在这里可能会有一个疑问，如果在声明的字符串内部有一个字符组合正好是)"，这样原生字符串不就会被截断了吗？没错，如果出现这样的情况，编译会出错。不过，我们也不必担心这种情况，C++11标准已经考虑到了这个问题，所以有了delimiter（分隔符）这个元素。delimiter可以是由除括号、反斜杠和空格以外的任何源字符构成的字符序列，长度至多为16个字符。通过添加delimiter可以改变编译器对原生字符串字面量范围的判定，从而顺利编译带有)"的字符串，例如：

```cpp
char hello_world_html[] = R"cpp(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial
scale=1, user-scalable=yes">
<title>Hello World!</title>
</head>
<body>
"(Hello World!)"
< / body >
< / html>
)cpp";
```

在上面的代码中，字符串虽然包含`"(Hello World!)"`这个比较特殊的子字符串，但是因为我们添加了cpp这个分隔符，所以编译器能正确地获取字符串的真实范围，从而顺利地通过编译。

C++11标准除了让我们能够定义char类型的原生字符串字面量外，对于`wchar_t`、`char8_t`（C++20标准开始）、`char16_t`和`char32_t`类型的原生字符串字面量也有支持。要支持这4种字符类型，就需要用到另外一个可选元素`prefix`了。这里的`prefix`实际上是声明4个类型字符串的前缀L、u、U和u8。

```cpp
char8_t utf8[] = u8R"(你好世界)"; // C++20标准开始
char16_t utf16[] = uR"(你好世界)";
char32_t utf32[] = UR"(你好世界)";
wchar_t wstr[] = LR"(你好世界)";
```

最后，关于原生字符串字面量的连接规则实际上和普通字符串字面量是一样的，唯一需要注意的是，原生字符串字面量除了能连接原生字符串字面量以外，还能连接普通字符串字面量。

## 用户自定义字面量

在C++11标准中新引入了一个用户自定义字面量的概念，程序员可以通过自定义后缀将整数、浮点数、字符和字符串转化为特定的对象。这个特性往往用在需要大量声明某个类型对象的场景中，它能够减少一些重复类型的书写，避免代码冗余。一个典型的例子就是不同单位对象的互相操作，比如长度、重量、时间等，举个例子：

```cpp
#include <iostream>

template<int scale, char ... unit_char>
struct LengthUnit {
	constexpr static int value = scale;
	constexpr static char unit_str[sizeof...(unit_char) + 1] = { unit_char..., '\0'};
};

template<class T>
class LengthWithUnit {
public:
	LengthWithUnit() : length_unit_(0) {}
	LengthWithUnit(unsigned long long length) : length_unit_(length* T::value) {}
	template<class U>
	LengthWithUnit<std::conditional_t<(T::value > U::value), U, T>> operator+(const LengthWithUnit<U> &rhs) {
		using unit_type = std::conditional_t<(T::value > U::value), U, T>;
		return LengthWithUnit<unit_type>((length_unit_ + rhs.get_length()) / unit_type::value);
	}
	unsigned long long get_length() const { return length_unit_; }
	constexpr static const char* get_unit_str() { return T::unit_str; }

private:
	unsigned long long length_unit_;
};

template<class T>
std::ostream& operator<< (std::ostream& out, const LengthWithUnit<T> &unit) {
	out << unit.get_length() / T::value << LengthWithUnit<T>::get_unit_str();
	return out;
}

using MMUnit = LengthUnit<1, 'm', 'm'>;
using CMUnit = LengthUnit<10, 'c', 'm'>;
using DMUnit = LengthUnit<100, 'd', 'm'>;
using MUnit = LengthUnit<1000, 'm'>;
using KMUnit = LenggthUnit<1000000, 'k', 'm'>;

using LengthWithMMUnit = LengthWithUnit<MMUnit>;
using LengthWithCMUnit = LengthWithUnit<CMUnit>;
using LengthWithDMUnit = LengthWithUnit<DMUnit>;
using LengthWithMUnit = LengthWithUnit<MUnit>;
using LengthWithKMUnit = LengthWithUnit<KMUnit>;

int main() {
	auto total_length = LengthWithCMUnit(1) + LengthWithMUnit(2) + LengthWithMMUnit(4);
	std::cout << total_length;
}
```



# `alignas`和`alignof`（C++11C++17）

## 不可忽视的数据对齐问题

C++11中新增了`alignof`和`alignas`两个关键字，其中`alignof`运算符可以用于获取类型的对齐字节长度，`alignas`说明符可以用来改变类型的默认对齐字节长度。这两个关键字的出现解决了长期以来C++标准中无法对数据对齐进行处理的问题。

在详细介绍这两个关键字之前，我们先来看一看下面这段代码：

```cpp
#include <iostream>

struct A {
	char a1;
	int a2;
	double a3;
};

struct B {
	short b1;
	bool b2;
	double b3;
};

int main() {
	std::cout << "sizeof(A::a1) + sizeof(A::a2) + sizeof(A::a3) = " << sizeof(A::a1) + sizeof(A::a2) + sizeof(A::a3) << std::endl;
	std::cout << "sizeof(B::b1) + sizeof(B::b2) + sizeof(B::b3) = " << sizeof(B::b1) + sizeof(B::b2) + sizeof(B::b3) << std::endl;
	std::cout << "sizeof(A) = " << sizeof(A) << std::endl;
	std::cout << "sizeof(B) = " << sizeof(B) << std::endl;
}
```

编译运行这段代码会得到以下结果：

```txt
sizeof(A::a1) + sizeof(A::a2) + sizeof(A::a3) = 13
sizeof(B::b1) + sizeof(B::b2) + sizeof(B::b3) = 11
sizeof(A) = 16
sizeof(B) = 16
```

