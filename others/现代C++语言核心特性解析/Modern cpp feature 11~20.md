---
date: 2024-11-24 22:58:54
date modified: 2024-11-27 16:18:15
title: Modern cpp feature 11~20
tags:
  - cpp
categories:
  - cpp
date created: 2024-09-25 13:26:08
---
Reference: 《现代C++语言核心特性解析》

11. 非受限联合类型 (C++ 11)

12. 委托构造函数

13. 继承构造函数

14. 强枚举类型

15. 扩展的聚合类型

16. `override`和`final`说明符（C++11）

17. 基于范围的for循环（C++11 C++17 C++20）

18. 支持初始化语句的 `if` 和 `switch`（C++17）

19. `static_assert` 声明

20. 结构化绑定（C++17 C++20）

<!-- more -->

# 非受限联合类型

## 联合类型在C++中的局限性

在编程的问题中，用尽量少的内存做尽可能多的事情一直都是一个重要的课题。C++中的联合类型（union）可以说是节约内存的一个典型代表。因为在联合类型中多个对象可以共享一片内存，相应的这片内存也只能由一个对象使用，例如：

```cpp
#include <iostream>

union U {
	int x1;
	float x2;
};

int main() {
	U u;
	u.x1 = 5;
	std::cout << u.x1 << std::endl;
	std::cout << u.x2 << std::endl;
	
	u.x2 = 5.0;
	std::cout << u.x1 << std::endl;
	std::cout << u.x2 << std::endl;
}
```

令人遗憾的是，过去的联合类型在C++中的使用并不广泛，因为C++中的大多数对象不能成为联合类型的成员。**过去的C++标准规定，联合类型的成员变量的类型不能是一个非平凡类型，也就是说它的成员类型不能有自定义构造函数**，比如：

```cpp
union U {
	int x1;
	float x2;
	std::string x3;
};
```

## 使用非受限联合类型

在C++11中如果有联合类型中存在非平凡类型，那么这个联合类型的特殊成员函数将被隐式删除，也就是说**我们必须自己至少提供联合类型的构造和析构函数**，比如：

```cpp
#include <iostream>
#include <string>
#include <vector>

union U {
	U() {} // 存在非平凡类型成员，必须提供构造函数
	~U() {} // 存在非平凡类型成员，必须提供析构函数
	int x1;
	float x2;
	std::string x3;
	std::vector<int> x4;
};

int main() {
	U u;
	u.x3 = "helloworld";
	std::cout << u.x3;
}
```

不过请注意，能够编译通过并不代表没有问题，实际上这段代码会运行出错，因为非平凡类型x3并没有被构造，所以在赋值操作的时候必然会出错。现在修改一下代码：

```cpp
#include <iostream>
#include <string>
#include <vector>

union U {
	U() : x3() {}
	~U() { x3.~basic_string(); }
	int x1;
	float x2;
	std::string x3;
	std::vector<int> x4;
};

int main() {
	U u;
	u.x3 = "hello world";
	std::cout << u.x3;
}
```



```cpp
#include <iostream>
#include <string>
#include <vector>

union U {
	U() : x3() {}
	~U() { x3.~basic_string(); }
	int x1;
	float x2;
	std::string x3;
	std::vector<int> x4;
};

int main() {
	U u;
	u.x4.push_back(58); 
}
```

基于这些考虑，我还是比较推荐让联合类型的构造和析构函数为空，也就是什么也不做，并且将其成员的构造和析构函数放在需要使用联合类型的地方。让我们继续修改上面的代码：

```cpp
#include <iostream>
#include <string>
#include <vector>

union U {
	U() {}
	~U() {}
	int x1;
	float x2;
	std::string x3;
	std::vector<int> x4;
};

int main() {
	U u;
	new(&u.x3) std::string("hello world");
	std::cout << u.x3 << std::endl;
	u.x3.~basic_string();

	new(&u.x4) std::vector<int>;
	u.x4.push_back(58);
	std::cout << u.x4[0] << std::endl;
	u.x4.~vector();
}
```

请注意，上面的代码用了`placement new`的技巧来初始化构造`x3`和`x4`对象，在使用完对象后手动调用对象的析构函数。通过这样的方法保证了联合类型使用的灵活性和正确性。

最后简单介绍一下非受限联合类型对静态成员变量的支持。联合类型的静态成员不属于联合类型的任何对象，所以并不是对象构造时被定义的，不能在联合类型内部初始化。实际上这一点和类的静态成员变量是一样的，当然了，它的初始化方法也和类的静态成员变量相同：

```cpp
#include <iostream>

union U {
	static int x1;
};
int U::x1 = 42;

int main() {
	std::cout << U::x1 << std::endl;
}
```

# 委托构造函数

## 冗余的构造函数

一个类有多个不同的构造函数在C++中是很常见的，例如：

```cpp
class X {
public:
	X() : a_(0), b_(0.) { CommonInit(); }
	X(int a) : a_(a), b_(0.) { CommonInit(); }
	X(double b) : a_(0), b_(b) { CommonInit(); }
	X(int a, double b) : a_(a), b_(b) { CommonInit(); }
private:
	void CommonInit() {}
	int a_;
	double b_;
};
```

也许有读者会提出将数据成员的初始化放到CommonInit函数里，从而减轻初始化列表代码冗余的问题，例如：

```cpp
class X1 {
public:
	X1() { CommonInit(0, 0.); }
	X1(int a) { CommonInit(a, 0.); }
	X1(double b) { CommonInit(0, b); }
	X1(int a, double b) { CommonInit(a, b); }
private:
	void CommonInit(int a, double b) {
		a_ = a;
		b_ = b;
	}
	int a_;
	double b_;
};
```

如果成员函数中包含复杂的对象，那么就可能引发不确定问题，最好的情况是只影响类的构造效率，例如：

```cpp
class X2 {
public:
	X2() { CommonInit(0, 0.); }
	X2(int a) { CommonInit(a, 0.); }
	X2(double b) { CommonInit(0, b); }
	X2(int a, double b) { CommonInit(a, b); }
private:
	void CommonInit(int a, double b) {
		a_ = a;
		b_ = b;
		c_ = "hello world";
	}
	int a_;
	double b_;
	std::string c_;
};
```

在上面的代码中，`std::string`类型的对象`c_`看似是在`CommonInit`函数中初始化为`hello world`，但是实际上它并不是一个初始化过程，而是一个赋值过程。**因为对象的初始化过程早在构造函数主体执行之前，也就是初始化列表阶段就已经执行了。** 所以这里的c_对象进行了两次操作，一次为初始化，另一次才是赋值为`hello world`，很明显这样对程序造成了不必要的性能损失。另外，有些情况是不能使用函数主体对成员对象进行赋值的，比如禁用了赋值运算符的数据成员。

当然读者还可能会提出通过为构造函数提供默认参数的方法来解决代码冗余的问题，例如：

```cpp
class X3 {
public:
	X3(double b) : a_(0), b_(b) { CommonInit(); }
	X3(int a = 0, double b = 0.) : a_(a), b_(b) { CommonInit(); }
private:
	void CommonInit() {}
	int a_;
	double b_;
};
```

这种做法的作用非常有限，可以看到上面这段代码，虽然通过默认参数的方式优化了两个构造函数，但是对于X3(double b)这个构造函数依然需要在初始化列表中重复初始化成员变量。另外，使用默认参数稍不注意就会引发二义性的问题，例如

```cpp
class X4 {
public:
	X4(int c) : a_(0), b_(0.), c_(c) { CommonInit(); }
	X4(double b) : a_(0), b_(b), c_(0) { CommonInit(); }
	X4(int a = 0, double b = 0., int c = 0) : a_(a), b_(b), c_(c) { CommonInit(); }
private:
	void CommonInit() {}
	int a_;
	double b_;
	int c_;
};

int main() {
	X4 x4(1);
}
```

以上代码无法通过编译，因为当main函数对x4进行构造时，编译器不知道应该调用`X4(int c)`还是`X4(int a = 0, double b= 0., int c = 0)`。所以让构造函数使用默认参数也不是一个好的解决方案。

## 委托构造函数

为了合理复用构造函数来减少代码冗余，C++11标准支持了委托构造函数：**某个类型的一个构造函数可以委托同类型的另一个构造函数对对象进行初始化。** 为了描述方便我们称前者为委托构造函数，后者为代理构造函数（英文直译为目标构造函数）。委托构造函数会将控制权交给代理构造函数，在代理构造函数执行完之后，再执行委托构造函数的主体。委托构造函数的语法非常简单，只需要在委托构造函数的初始化列表中调用代理构造函数即可，例如：

```cpp
class X {
public:
	X() : X(0, 0.) {}
	X(int a) : X(a, 0.) {}
	X(double b) : X(0, b) {}
	X(int a, double b) : a_(a), b_(b) { CommonInit(); }
private:
	void CommonInit() {}
	int a_;
	double b_;
};
```

可以看到X()、X(int a)、X(double b)分别作为委托构造函数将控制权交给了代理构造函数X(int a, double b)。它们的执行顺序是先执行代理构造函数的初始化列表，接着执行代理构造函数的主体（也就是CommonInit函数），最后执行委托构造函数的主体，在这个例子中委托构造函数的主体都为空。

委托构造函数的语法很简单，不过想合理使用它还需注意以下5点。

**每个构造函数都可以委托另一个构造函数为代理。也就是说，可能存在一个构造函数，它既是委托构造函数也是代理构造函数，例如：**

```cpp
class X {
public:
	X() : X(0) {}
	X(int a) : X(a, 0.) {}
	X(double b) : X(0, b) {}
	X(int a, double b) : a_(a), b_(b) { CommonInit(); }
private:
	void CommonInit() {}
	int a_;
	double b_;
};
```

在上面的代码中构造函数`X(int a)`，它既是一个委托构造函数，也是`X()`的代理构造函数。另外，除了自定义构造函数以外，我们还能让特殊构造函数也成为委托构造函数，例如：

```cpp
class X {
public:
	X() : X(0) {}
	X(int a) : X(a, 0.) {}
	X(double b) : X(0, b) {}
	X(int a, double b) : a_(a), b_(b) { CommonInit(); }
	X(const X& other) : X(other.a_, other.b_) {}  // 委托复制构造函数
private:
	void CommonInit() {}
	int a_;
	double b_;
};
```

**不要递归循环委托！** 这一点非常重要，因为循环委托不会被编译器报错，随之而来的是程序运行时发生未定义行为，最常见的结果是程序因栈内存用尽而崩溃：

```cpp
class X {
public:
	X() : X(0) {}
	X(int a) : X(a, 0.) {}
	X(double b) : X(0, b) {}
	X(int a, double b) : X() { CommonInit(); }
private:
	void CommonInit() {}
	int a_;
	double b_;
};
```

**如果一个构造函数为委托构造函数，那么其初始化列表里就不能对数据成员和基类进行初始化：**

```cpp
class X {
public:
	X() : a_(0), b_(0) { CommonInit(); }
	X(int a) : X(), a_(a) {} // 编译错误，委托构造函数不能在初始化列表初始化成员变量
	X(double b) : X(), b_(b) {} // 编译错误，委托构造函数不能在初始化列表初始化成员变量

private:
	void CommonInit() {}
	int a_;
	double b_;
};
```

**委托构造函数的执行顺序是先执行代理构造函数的初始化列表，然后执行代理构造函数的主体，最后执行委托构造函数的主体，** 例如：

```cpp
#include <iostream>

class X {
public:
	X() : X(0) { InitStep3(); }
	X(int a) : X(a, 0.) { InitStep2(); }
	X(double b) : X(0, b) {}
	X(int a, double b) : a_(a), b_(b) { InitStep1(); }
private:
	void InitStep1() { std::cout << "InitStep1()" << std::endl; }
	void InitStep2() { std::cout << "InitStep2()" << std::endl; }
	void InitStep3() { std::cout << "InitStep3()" << std::endl; }
	int a_;
	double b_;
};

int main() {
	X x;
}
```

编译执行以上代码，输出结果如下：

```txt
InitStep1()
InitStep2()
InitStep3()
```

**如果在代理构造函数执行完成后，委托构造函数主体抛出了异常，则自动调用该类型的析构函数。** C++标准规定（规则3也提到过），一旦类型有一个构造函数完成执行，那么就会认为其构造的对象已经构造完成，所以发生异常后需要调用析构函数，来看一看具体的例子：

```cpp
#include <iostream>

class X {
	X() : X(0, 0.) { throw 1; }
	X(int a) : X(a, 0.) {}
	X(double b) : X(0, b) {}
	X(int a, double b) : a_(a), b_(b) { CommonInit(); }
	~X() { std::cout << "~X()" << std::endl; }
private:
	void CommonInit() {}
	int a_;
	double b_;
};

int main() {
	try {
		X x;
	} catch(...) {
	
	}
}
```

## 委托模板构造函数

委托模板构造函数是指一个构造函数将控制权委托到同类型的一个模板构造函数，简单地说，就是代理构造函数是一个函数模板。这样做的意义在于泛化了构造函数，减少冗余的代码的产生。将代理构造函数编写成函数模板往往会获得很好的效果，让我们看一看例子：

```cpp
#include <vector>
#include <list>
#include <deque>

class X {
	template<class T> X(T first, T last) : l_(first, last) {}
	std::list<int> l_;
public:
	X(std::vector<short>&);
	X(std::deque<int>&);
};
X::X(std::vector<short>& v) : X(v.begin(), v.end()) {}
X::X(std::deque<int>& v) : X(v.begin(), v.end()) {}

int main() {
	std::vector<short> a{ 1, 2, 3, 4, 5 };
	std::deque<int> b{ 1, 2, 3, 4, 5 };
	X x1(a);
	X x2(b);
}
```

在上面的代码中`template<class T> X(T first, Tlast)`是一个代理模板构造函数，`X(std::vector<short>&)`和`X(std::deque<int>&)`将控制权委托给了它。这样一来，我们就无须编写`std::vector<short>`和`std::deque <int>`版本的代理构造函数。后续增加委托构造函数也不需要修改代理构造函数，只需要保证参数类型支持迭代器就行了。

## 捕获委托构造函数的异常

```cpp
#include <iostream>

class X {
public:
	X() try : X(0) {}
	catch (int e) {
		std::cout << "catch: " << e << std::endl;
		throw 3;
	}
	
	X(int a) try : X(a, 0.) {}
	catch (int e) {
		std::cout << "catch: " << e << std::endl;
		throw 2;
	}
	X(double b) : X(0, b) {}
	X(int a, double b) : a_(a), b_(b) { throw 1; }
private:
	int a_;
	double b_;
};

int main() {
	try {
		X x;
	} catch (int e) {
		std::cout << "catch: " << e << std::endl;
	}
}
```

编译运行以上代码，输出结果如下：

```txt
catch: 1
catch: 2
catch: 3
```

## 委托参数较少的构造函数

将参数较少的构造函数委托给参数较多的构造函数。通常情况下我们建议这么做，因为这样做的自由度更高。但是，并不是完全否定从参数较多的构造函数委托参数较少的构造函数的意义。这种情况通常发生在构造函数的参数必须在函数体中使用的场景。以`std::fstream`作为例子：

```cpp
basic_fstream();
explicit basic_fstream(const char* s, ios_base::openmode mode);
```

`basic_fstream`的这两个构造函数，由于`basic_fstream(const char * s, ios_base::openmode mode)`需要在构造函数体内执行具体打开文件的操作，所以它完全可以委托`basic_fstream()`来完成一些最基础的初始化工作，最后执行到自己的主体时再打开文件:

```cpp
basic_fstream::basic_fstream(const char* s, ios_base::openmode mode) : basic_fstream() {
	if (open(s, mode) == 0) 
		setstate(failbit);
}
```

# 继承构造函数

## 继承关系中构造函数的困局

假设现在有一个类`Base`提供了很多不同的构造函数。某一天，你发现`Base`无法满足未来业务需求，需要把`Base`作为基类派生出一个新类`Derived`并且对某些函数进行改造以满足未来新的业务需求，比如下面的代码：

```cpp
class Base {
public:
	Base() : x_(0), y_(0.) {};
	Base(int x, double y) : x_(x), y_(y) {}
	Base(double y) : x_(0), y_(y) {}
	void SomeFunc() {}
private:
	int x_;
	double y_;
};

class Derived : public Base {
public:
	Derived() {};
	Derived(int x, double y) : Base(x, y) {}
	Derived(int x) : Base(x) {}
	Derived(double y) : Base(y) {}
	void SomeFunc() {}
};
```

基类`Base`的`SomeFunc`无法满足当前的业务需求，于是在其派生类`Derived`中重写了这个函数，但令人头痛的是，面对`Base`中大量的构造函数，我们不得不在`Derived`中定义同样多的构造函数，目的仅仅是转发构造参数，因为派生类本身并没有需要初始化的数据成员。单纯地转发构造函数不仅会导致代码的冗余，而且大量重复的代码也会让程序更容易出错。实际上，这个工作完全可以让编译器自动完成，因为它实在太简单了，让编译器代劳不仅消除了代码冗余而且意图上也更加明确。

## 使用继承构造函数

我们都知道C++中可以使用using关键字将基类的函数引入派生类，比如：

```cpp
class Base {
public:
	void foo(int) {}
};

class Derived : public Base {
public:
	using Base::foo;
	void foo(char*) {}
};

int main() {
	Derived d;
	d.foo(5);
}
```

C++11的继承构造函数正是利用了这一点，将using关键字的能力进行了扩展，使其能够引入基类的构造函数：

```cpp
class Base {
public:
	Base() : x_(0), y_(0.) {};
	Base(int x, double y) : x_(x), y_(y) {}
	Base(int x) : x_(x), y_(0.) {}
	Base(double y) : x_(0), y_(y) {}
private:
	int x_;
	double y_;
}

class Derived : public Base {
public:
	using Base::Base;
};
```

使用继承构造函数虽然很方便，但是还有6条规则需要注意。

**派生类是隐式继承基类的构造函数，所以只有在程序中使用了这些构造函数，编译器才会为派生类生成继承构造函数的代码。**

**派生类不会继承基类的默认构造函数和复制构造函数。** 这一点乍看有些奇怪，但仔细想想也是顺理成章的。因为在C++语法规则中，执行派生类默认构造函数之前一定会先执行基类的构造函数。同样的，在执行复制构造函数之前也一定会先执行基类的复制构造函数。所以继承基类的默认构造函数和默认复制构造函数的做法是多余的，这里不会这么做。

**继承构造函数不会影响派生类默认构造函数的隐式声明，也就是说对于继承基类构造函数的派生类，编译器依然会为其自动生成默认构造函数的代码。**

**在派生类中声明签名相同的构造函数会禁止继承相应的构造函数。** 这一条规则不太好理解，让我们结合代码来看一看：

```cpp
class Base {
public:
	Base() : x_(0), y_(0.) {}
	Base(int x, double y) : x_(x), y_(y) {}
	Base(int x) : x_(x), y_(0.) { std::cout << "Base(int x)" << std::endl; }
	Base(double y) : x_(0), y_(y) { std::cout << "Base(double y)" << std::endl; }
private:
	int x_;
	double y_;
};

class Derived : public Base {
public:
	using Base::Base;
	Derived(int x) { str::cout << "Derived(int x)" << std::endl; }
};

int main() {
	Derived d(5);
	Derived d1(5.5);
}
```

在上面的代码中，派生类`Derived`使用`using Base::Base`继承了基类的构造函数，但是由于`Derived`定义了构造函数`Derived(int x)`，该函数的签名与基类的构造函数`Base(int x)`相同，因此这个构造函数的继承被禁止了，`Derived d(5)`会调用派生类的构造函数并且输出"Derived(int x)"。另外，这个禁止动作并不会影响到其他签名的构造函数，`Derived d1(5.5)`依然可以成功地使用基类的构造函数进行构造初始化。

**派生类继承多个签名相同的构造函数会导致编译失败：**

```cpp
class Base1 {
public:
	Base1(int) { std::cout << "Base1(int x)" << std::endl; };
};

class Base2 {
public:
	Base2(int) { std::cout << "Base2(int x)" << std::endl; };
};

class Derived : public Base1, Base2 {
public:
	using Base1::Base1;
	using Base2::Base2;
};

int main() {
	Derived d(5);
}
```

在上面的代码中，`Derived`继承了两个类`Base1`和`Base2`，并且继承了它们的构造函数。但是由于这两个类的构造函数`Base1(int)`和`Base2(int)`拥有相同的签名，导致编译器在构造对象的时候不知道应该使用哪一个基类的构造函数，因此在编译时给出一个二义性错误。

**继承构造函数的基类构造函数不能为私有：**

```cpp
class Base {
	Base(int) {}
public:
	Base(double) {}
};

class Derived : public Base {
public:
	using Base::Base;
};

int main() {
	Derived d(5.5);
	Derived d1(5);
}
```

在上面的代码中，Derived d1(5)无法通过编译，因为它对应的基类构造函数Base(int)是一个私有函数，Derived d(5.5)则没有这个问题。

最后再介绍一个有趣的问题，在早期的C++11编译器中，继承构造函数会把基类构造函数注入派生类，于是导致了这样一个问题：

```cpp
#include <iostream>

struct Base {
	Base() = default;
	template<typename T> Base(T, typename T::type=0) {
		std::cout << "Base(T, typename T::type)" << std::endl;
	}
	Base(int) { std::cout << "Base(int)" << std::endl; }
};

struct Derived : Base {
	using Base::Base;
	Derived(int) { std::cout << "Derived(int)" << std::endl; }
};

int main() {
	Derived d(42L);
}
```

上面这段代码用早期的编译器（比如GCC 6.4）编译运行的输出结果是`Base(int)`，而用新的GCC编译运行的输出结果是`Derived(int)`。在老的版本中，`template<typename T>Base(T, typename T::type = 0)` 被注入派生类中，形成了这样两个构造函数：

```cpp
template<typename T> Derived(T);
template<typename T> Derived(T, typename T::type);
```

这是因为继承基类构造函数时，不会继承默认参数，而是在派生类中注入带有各种参数数量的构造函数的重载集合。于是，编译器理所当然地选择推导`Derived(T)`为`Derived(long)`作为构造函数。在构造基类时，由于`Base(long, typename long::type = 0)`显然是一个非法的声明，因此编译器选择使用`Base(int)`作为基类的构造函数。最终结果就是我们看到的输出了`Base(int)`。而在新版本中继承构造函数不会注入派生类，所以不存在这个问题，编译器会直接使用派生类的`Derived(int)`构造函数构造对象。

# 强枚举类型

大多数情况下，我们说C++是一门类型安全的强类型语言，但是枚举类型在一定程度上却是一个例外，具体来说有以下几个方面的原因。

首先，虽然枚举类型存在一定的安全检查功能，一个枚举类型不允许分配到另外一种枚举类型，而且整型也无法隐式转换成枚举类型。但是枚举类型却可以隐式转换为整型，因为C++标准文档提到“枚举类型可以采用整型提升的方法转换成整型”。请看下面的代码示例：

```cpp
enum School {
	principal,
	teacher,
	student
};

enum Company {
	chairman,
	manager,
	employee
};

int main() {
	School x = student;
	Company y = manager;
	bool b = student >= manager; // 不同类型之间的比较操作
	b = x < employee;
	int y = student; // 隐式转换为int
}
```

在上面的代码中两个不同类型的枚举标识符student和manager可以进行比较，这在C++语言的其他类型中是很少看到的。这种比较合法的原因是枚举类型先被隐式转换为整型，然后才进行比较。同样的问题也出现在student直接赋值到int类型变量上的情况中。另外，下面的代码会触发C++对枚举的检查，它们是无法编译通过的：

```cpp
School x = chairman;    // 类型不匹配，无法通过编译
Company y = student;    // 类型不匹配，无法通过编译
x = 1;                  // 整型无法隐式转换到枚举类型      
```

然后是枚举类型的作用域问题，枚举类型会把其内部的枚举标识符导出到枚举被定义的作用域。也是就说，我们使用枚举标识符的时候，可以跳过对于枚举类型的描述：

```cpp
School x = student;
Company y = manager;
```

无论是初始化x，还是初始化y，我们都没有对student和manager的枚举类型进行描述。因为它们已经跳出了School和Company。在我们看到的第一个例子中，这没有什么问题，两种类型相安无事。但是如果遇到下面的这种情况就会让人头痛了：

```cpp
enum HighSchool {
	student;
	teacher;
	principal;
};

enum University {
	student;
	professor,
	principal
};
```

HighSchool和University都有student和principal，而枚举类型又会将其枚举标识符导出到定义它们的作用域，这样就会发生重复定义，无法通过编译。解决此类问题的一个办法是使用命名空间，例如：

```cpp
enum HighSchool {
	student,
	teacher,
	principal
};

namespace AcademicInstitution {
	enum University {
		student,
		professor,
		principal
	};
}
```

对于上面两个问题，有一个比较好但并不完美的解决方案，代码如下：

```cpp
#include <iostream>

class AuthorityType {
	enum InternalType {
		ITBan, 
		ITGuest,
		ITMember,
		ITAdmin,
		ITSystem,
	};
	InternalType self_;

public:
	AuthorityType(InternalType self) : self_(self) {}
	bool operator < (const AuthorityType &other) const {
		return self_ < other.self_;
	}
	bool operator > (const AuthorityType &other) const {
		return self_ > other.self_;
	}
	bool operator <= (const AuthorityType &) const {
		return self_ <= other.self_;
	}
	bool operator >= (const AutorityType &) const {
		return self_ >= other.self_;
	}
	bool operator == (const AuthorityType &) const {
		return self_ == other.self_;
	}
	bool operator != (const AuthorityType& other) const {
		return self_ != other.self_;
	}
	const static AuthorityType System, Admin, Member, Guest, Ban;
};

#define DEFINE_AuthorityType(x) const AuthorityType \ AuthorityType::x(AuthorityType::IT ## x)
DEFINE_AuthorityType(System);
DEFINE_AuthorityType(Admin);
DEFINE_AuthorityType(Member);
DEFNIE_AuthorityType(Guest);
DEFINE_AuthorityType(Ban);

int main() {
	bool b = AuthorityType::System > AuthorityType::Admin;
	std::cout << std::boolalpha << b << std::endl;
}
```

将枚举类型变量封装成类私有数据成员，保证无法被外界访问。访问枚举类型的数据成员必须通过对应的常量静态对象。另外，根据C++标准的约束，访问静态对象必须指明对象所属类型。也就是说，如果我们想访问ITSystem这个枚举标识符，就必须访问常量静态对象System，而访问System对象，就必须说明其所属类型，这使我们需要将代码写成AuthorityType:: System才能编译通过。

由于我们实现了比较运算符，因此可以对枚举类型进行比较。但是比较运算符函数只接受同类型的参数，所以只允许相同类型进行比较。

最大的缺点是实现起来要多敲很多代码。枚举类型本身是一个POD类型，而我们实现的类破坏了这种特性。

还有一个严重的问题是，无法指定枚举类型的底层类型。因此，不同的编译器对于相同枚举类型可能会有不同的底层类型，甚至有无符号也会不同。来看下面这段代码：

```cpp
enum E {
	e1 = 1,
	e2 = 2,
	e3 = 0xfffffff0
};

int main() {
	bool b = e1 < e3;
	std::cout << std::boolalpha << b << std::endl;
}
```

如果代码中有需要表达枚举语义的地方，还是应该使用枚举类型。原因就是在第一个问题中讨论的，枚举类型还是有一定的类型检查能力。我们应该避免使用宏和const int的方法去实现枚举，因为其缺点更加严重。

值得一提的是，枚举类型缺乏类型检查的问题倒是成就了一种特殊用法。如果读者了解模板元编程，那么肯定见过一种被称为`enum hack`的枚举类型的用法。简单来说就是利用枚举值在编译期就能确定下来的特性，让编译器帮助我们完成一些计算.

```cpp
#include <iostream>
template<int a, int b>
struct add {
	enum {
		result = a + b
	};
};

int main() {
	std::cout << add<5, 8>::result << std::endl;
}
```

用 GCC 查看其GIMPLE的中间代码

```cpp
main() {
	int D.39267;
	_1 = std::basic_ostream<char>::operator<< (&cout, 13);
	std::basic_ostream<char>::operator<< (_1, endl);
	D.39267 = 0;
	return D.39267;
}
```

可以看到add<5, 8>::result在编译器编译代码的时候就已经计算出来了，运行时直接使用<<运算符输出结果13。

## 使用强枚举类型


由于枚举类型确实存在一些类型安全的问题，因此C++标准委员会在C++11标准中对其做出了重大升级，增加了强枚举类型。另外，为了保证老代码的兼容性，也保留了枚举类型之前的特性。强枚举类型具备以下3个新特性。

- 枚举标识符属于强枚举类型的作用域。

- 枚举标识符不会隐式转换为整型。

- 能指定强枚举类型的底层类型，底层类型默认为int类型。

定义强枚举类型的方法非常简单，只需要在枚举定义的enum关键字之后加上class关键字就可以了。下面将`HighSchool`和`University`改写为强枚举类型：

```cpp
#include <iostream>

enum class HighSchool {
	student,
	teacher,
	principal
};

enum class University {
	student,
	professor,
	principal
};

int main() {
	HighSchool x = HighSchool::student;
	University y = University::student;
	bool b = x < HighSchool::headmaster;
	std::cout << std::boolalpha << b << std::endl;
}
```

观察上面的代码可以发现，首先，在不使用命名空间的情况下，两个有着相同枚举标识符的强枚举类型可以在一个作用域内共存。这符合强枚举类型的第一个特性，其枚举标识符属于强枚举类型的作用域，无法从外部直接访问它们，所以在访问时必须加上枚举类型名，否则会编译失败，如`HighSchool::student`。其次，相同枚举类型的枚举标识符可以进行比较，但是不同枚举类型就无法比较其枚举标识符了，因为它们失去了隐式转换为整型的能力，这一点符合强枚举类型的第二个特性：

```cpp
HighSchool x = student; // 编译失败，找不到student的定义
bool b = University::student < HighSchool::headmaster; // 编译失败，比较的类型不同
int y = University::student;   // 编译失败，无法隐式转换为 int 类型
```

对于强枚举类型的第三个特性，我们可以在定义类型的时候使用:符号来指明其底层类型。利用它可以消除不同编译器带来的歧义：

```cpp
enum class E : unsigned int {
	e1 = 1,
	e2 = 2,
	e3 = 0xfffffff0
};

int main() {
	bool b = e1 < e3;
	std::cout << std::boolalpha << b << std::endl;
}
```

在C++11标准中，我们除了能指定强枚举类型的底层类型，还可以指定枚举类型的底层类型，例如：

```cpp
enum E : unsigned int {
	e1 = 1,
	e2 = 2,
	e3 = 0xfffffff0
};

int main() {
	bool b = e1 < e3;
	std::cout << std::boolalpha << b << std::endl;
}
```

另外，虽然我们多次强调了强枚举类型的枚举标识符是无法隐式转换为整型的，但还是可以通过`static_cast`对其进行强制类型转换，但我建议不要这样做。最后说一点，**强枚举类型不允许匿名**，我们必须给定一个类型名，否则无法通过编译。

## 列表初始化有底层类型枚举对象

从C++17标准开始，对有底层类型的枚举类型对象可以直接使用列表初始化。这条规则适用于所有的强枚举类型，因为它们都有默认的底层类型`int`，而枚举类型就必须显式地指定底层类型才能使用该特性：

```cpp
enum class Color {
	Red,
	Green,
	Blue
};

int main() {
	Color c{ 5 };      // 编译成功
	Color c1 = 5;      // 编译失败
	Color c2 = { 5 };  // 编译失败
	Color c3(5);       // 编译失败
}
```

从C++17标准开始，对有底层类型的枚举类型对象可以直接使用列表初始化。这条规则适用于所有的强枚举类型，因为它们都有默认的底层类型int，而枚举类型就必须显式地指定底层类型才能使用该特性：

```cpp
enum class Color1 : char {};
enum Color2 : short {};

int main() {
	Color1 c{ 7 };
	Color2 c1{ 11 };
	Color2 c2 = Color2 { 5 };
}
```

没有指定底层类型的枚举类型是无法使用列表初始化的，比如：

```cpp
enum Color3 {};

int main() {
	Color3 c{ 7 };
}
```

以上代码一定会编译报错，因为无论是C++17还是在此之前的标准，Color3都没有底层类型。同所有的列表初始化一样，它禁止缩窄转换，所以下面的代码也是不允许的：

```cpp
enum class Color1 : char {};

int main() {
	Color1 c{ 7.11 };
}
```

让有底层类型的枚举类型支持列表初始化的确有一个十分合理的动机。

现在假设一个场景，我们需要一个新整数类型，该类型必须严格区别于其他整型，也就是说不能够和其他整型做隐式转换，显然使用`typedef`的方法是不行的。另外，虽然通过定义一个类的方法可以到达这个目的，但是这个方法需要编写大量的代码来重载运算符，也不是一个理想的方案。所以，C++的专家把目光投向了有底层类型的枚举类型，其特性几乎完美地符合以上要求，除了初始化整型值的时候需要用到强制类型转换。于是，C++17为有底层类型的枚举类型放宽了初始化的限制，让其支持列表初始化：

```cpp
#include <iostream>
enum class Index : int {};

int main() {
	Index a{ 5 };
	Index b{ 10 };
	// a = 12;
	// int c = b;
	std::cout << "a < b is " << std::boolaplha << (a < b) << std::endl;
}
```

在上面的代码中，定义了Index的底层类型为int，所以可以使用列表初始化a和b，由于a和b的枚举类型相同，因此所有a < b的用法也是合法的。但是a = 12和int c = b无法成功编译，因为强枚举类型是无法与整型隐式相互转换的。

最后提示一点，在C++17的标准库中新引入的std::byte类型就是用这种方法定义的。

## 使用`using`打开强枚举类型

C++20标准扩展了using功能，它可以打开强枚举类型的命名空间。在一些情况下，这样做会让代码更加简洁易读，例如：

```cpp
enum class Color {
	Red,
	Green,
	Blue
};

const char* ColorToString(Color c) {
	switch (c) {
		case Color::Red: return "Red";
		case Color::Green: return "Green";
		case Color::Blue: return "Blue";
		default:
			return "none"; 
	}
}
```

在上面的代码中，函数`ColorToString`中需要不断使用`Color::`来指定枚举标识符，这显然会让代码变得冗余。通过`using`我们可以简化这部分代码：

```cpp
const char* ColorToString(Color c) {
	switch (c) {
		using enum Color;
		case Red: return "Red";
		case Green: return "Green";
		case Blue: return "Blue";
		default:
			return "none";
	}
}
```

以上代码使用`using enum Color;`将`Color`中的枚举标识符引入`swtich-case`作用域。请注意，`swtich-case`作用域之外依然需要使用`Color::`来指定枚举标识符。除了引入整个枚举标识符之外，`using`还可以指定引入的标识符，例如：

```cpp
const char* ColorToString(Color c) {
	switch(c) {
		using Color::Red;
		case Red: return "Red";
		case Color::Green: return "Green";
		case Color::Blue: return "Blue";
		default:
			return "none";
	}
}
```

以上代码使用`using Color::Red;`将`Red`引入`swtich-case`作用域，其他枚举标识符依然需要使用Color::来指定。

# 扩展的聚合类型

C++17标准对聚合类型的定义做出了大幅修改，即从基类公开且非虚继承的类也可能是一个聚合。同时聚合类型还需要满足常规条件。

- 没有用户提供的构造函数。

- 没有私有和受保护的非静态数据成员。

- 没有虚函数。

在新的扩展中，如果类存在继承关系，则额外满足以下条件。

- 必须是公开的基类，不能是私有或者受保护的基类。

- 必须是非虚继承。

请注意，这里并没有讨论基类是否需要是聚合类型，也就是说基类是否是聚合类型与派生类是否为聚合类型没有关系，只要满足上述5个条件，派生类就是聚合类型。在标准库`<type_traits>`中提供了一个聚合类型的甄别办法`is_aggregate`，它可以帮助我们判断目标类型是否为聚合类型：

```cpp
#include <iostream>
#include <string>

class MyString: public std::string {};

int main() {
	std::cout << "std::is_aggregate_v<std::string> = " << std::is_aggregate_v<std::string> << std::endl;
	std::cout << "std::is_aggregate_v<MyString> = " << std::is_aggregate_v<MyString> << std::endl;
}
```

在上面的代码中，先通过`std::is_aggregate_v`判断`std::string`是否为聚合类型，根据我们对`std::string`的了解，它存在用户提供的构造函数，所以一定是非聚合类型。然后判断类`MyString`是否为聚合类型，虽然该类继承了`std::string`，但因为它是公开继承且是非虚继承，另外，在类中不存在用户提供的构造函数、虚函数以及私有或者受保护的数据成员，所以`MyString`应该是聚合类型。编译运行以上代码，输出的结果也和我们判断的一致：

```cpp
std::is_aggregate_v<std::string> = 0;
std::is_aggregate_v<MyString> = 1;
```

## 聚合类型的初始化

由于聚合类型定义的扩展，聚合对象的初始化方法也发生了变化。过去要想初始化派生类的基类，需要在派生类中提供构造函数，例如：

```cpp
#include <iostream>
#include <string>

class MyStringWithIndex : public std::string {
public:
	MyStringWithIndex(const std::string& str, int idx) : std::string(str), index_(idx) {}
	int index_ = 0;
};

std::ostream& operator << (std::ostream &o, const MyStringWithIndex& s) {
	o << s.index_ << ":" << s.c_str();
	return o;
}

int main() {
	MyStringWithIndex s("hello world", 11);
	std::cout << s << std::endl;
}
```

在上面的代码中，为了初始化基类我们不得不为`MyStringWithIndex`提供一个构造函数，用构造函数的初始化列表来初始化`std::string`。现在，由于聚合类型的扩展，这个过程得到了简化。需要做的修改只有两点，第一是删除派生类中用户提供的构造函数，第二是直接初始化：

```cpp
#include <iostream>
#include <string>

class MyStringWithIndex: public std::string {
public:
	int index_== 0 ;
};

std::ostream& operator << (std::ostream &o, const MyStringWithIndex& s) {
	o << s.index_ << ":" << s.c_str();
	return o;
}

int main() {
	MyStringWithIndex s{ {"hello world"}, 11 };
	std::cout << s << std::endl;
	std::cout << s << std::endl;
}
```

删除派生类中用户提供的构造函数是为了让`MyStringWithIndex`成为一个C++17标准的聚合类型，而作为聚合类型直接使用大括号初始化即可。另外，如果派生类存在多个基类，那么其初始化的顺序与继承的顺序相同：

```cpp
#include <iostream>
#include <string>

class Count {
public:
	int Get() { return count_++； }
	int count_ = 0;
};

class MyStringWithIndex: public std::string, public Count {
public:
	int index_ = 0;
};

std::ostream& operator << (std::ostream &o, MyStringWithIndex& s) {
	o << s.index_ << ":" << s.Get() << ":" << s.c_str();
	return o;
}

int main() {
	MyStringWithIndex s{ "hello world", 7, 11 };
	std::cout << s << std::endl;
	std::cout << s << std::endl;
}
```

在上面的代码中，类`MyStringWithIndex`先后继承了`std::string`和`Count`，所以在初始化时需要按照这个顺序初始化对象。`{ "hello world", 7, 11}`中字符串`"hello world"`对应基类`std::string`，`7`对应基类`Count`，`11`对应数据成员 `index_`。

## 扩展聚合类型的兼容问题

虽然扩展的聚合类型给我们提供了一些方便，但同时也带来了一个兼容老代码的问题，请考虑以下代码：

```cpp
#include <iostream>
#include <string>

class BaseData {
	int data_;
public:
	int Get() { return data_; }
protected:
	BaseData() : data_(11) {}
};

class DerivedData : public BaseData {
public:
};

int main() {
	DerivedData d{};
	std::cout << d.Get() << std::endl;
}
```

以上代码使用C++11或者C++14标准可以编译成功，而使用C++17标准编译则会出现错误，主要原因就是聚合类型的定义发生了变化。在C++17之前，类`DerivedData`不是一个聚合类型，所以`DerivedData d{}`会调用编译器提供的默认构造函数。调用`DerivedData`默认构造函数的同时还会调用`BaseData`的构造函数。虽然这里`BaseData`声明的是受保护的构造函数，但是这并不妨碍派生类调用它。从C++17开始情况发生了变化，类`DerivedData`变成了一个聚合类型，以至于`DerivedData d{}`也跟着变成聚合类型的初始化，因为基类`BaseData`中的构造函数是受保护的关系，它不允许在聚合类型初始化中被调用，所以编译器无奈之下给出了一个编译错误。如果读者在更新开发环境到C++17标准的时候遇到了这样的问题，**只需要为派生类提供一个默认构造函数即可。**

## 禁止聚合类型使用用户声明的构造函数

在前面我们提到没有用户提供的构造函数是聚合类型的条件之一，但是请注意，用户提供的构造函数和用户声明的构造函数是有区别的，比如：

```cpp
#include <iostream>
struct X {
	X() = default;
};

struct Y {
	Y() = delete;
};

int main() {
	std::cout << std::boolalpha << "std::is_aggregate_v<X> : " << std::is_aggregate_v<X> << std::endl << "std::is_aggregate_v<Y> : " << std::is_aggregate_v<Y> << std::endl; 
}
```

用C++17标准编译运行以上代码会输出：

```cpp
std::is_aggregate_v<X> : true
std::is_aggregate_v<Y> : true
```

由此可见，虽然类X和Y都有用户声明的构造函数，但是它们依旧是聚合类型。不过这就引出了一个问题，让我们将目光放在结构体Y上，因为它的默认构造函数被显式地删除了，所以该类型应该无法实例化对象，例如：

```cpp
Y y1; // 编译失败，使用了删除函数
```

但是作为聚合类型，我们却可以通过聚合初始化的方式将其实例化：

```cpp
Y y2{}; // 编译成功
```

编译成功的这个结果显然不是类型`Y`的设计者想看到的，而且这个问题很容易在真实的开发过程中被忽略，从而导致意想不到的结果。除了删除默认构造函数，将其列入私有访问中也会有同样的问题，比如：

```cpp
struct Y {
private:
	Y() = default;
};

Y y1; // 编译失败，构造函数为私有访问
y y2{} // 编译成功
```

请注意，这里`Y() = default;`中的`= default`不能省略，否则`Y`会被识别为一个非聚合类型。

为了避免以上问题的出现，在C++17标准中可以使用`explicit`说明符或者将`= default`声明到结构体外，例如：

```cpp
struct X {
	explicit X() = default;
};

struct Y {
	Y();
};
Y::Y() = default;
```

这样一来，结构体`X`和`Y`被转变为非聚合类型，也就无法使用聚合初始化了。不过即使这样，还是没有解决相同类型不同实例化方式表现不一致的尴尬问题，所以在C++20标准中禁止聚合类型使用用户声明的构造函数，这种处理方式让所有的情况保持一致，是最为简单明确的方法。同样是本节中的第一段代码示例，用C++20环境编译的输出结果如下：

```cpp
std::is_aggregate_v<X> : false 
std::is_aggregate_v<Y> : false
```

值得注意的是，这个规则的修改会改变一些旧代码的意义，比如我们经常用到的禁止复制构造的方法：

```cpp
struct X {
	std::string s;
	std::vector<int> v;
	X() = default;
	X(const X&) = delete;
	X(X &&) = default;
}
```

上面这段代码中结构体`X`在C++17标准中是聚合类型，所以可以使用聚合类型初始化对象。但是升级编译环境到C++20标准会使`X`转变为非聚合对象，从而造成无法通过编译的问题。一个可行的解决方案是，不要直接使用`= delete;`来删除复制构造函数，而是通过加入或者继承一个不可复制构造的类型来实现类型的不可复制，例如：

```cpp
struct X {
	std::string s;
	std::vector<int> v;
	[[no_unique_address]] NonCopyable nc;
};

// 或者

struct X : NonCopyable {
	std::string s;
	std::vector<int> v;
};
```

这种做法能让代码看起来更加简洁，所以我们往往会被推荐这样做。

## 使用带小括号的列表初始化聚合类型对象

通过15.2节，我们知道对于一个聚合类型可以使用带大括号的列表对其进行初始化，例如：

```cpp
struct X {
	int i;
	float f;
};

X x{ 11, 7.0f };
```

如果将上面初始化代码中的大括号修改为小括号，C++17标准的编译器会给出无法匹配到对应构造函数`X::X(int, float)`的错误，这说明小括号会尝试调用其构造函数。这一点在C++20标准中做出了修改，它规定对于聚合类型对象的初始化可以用小括号列表来完成，其最终结果与大括号列表相同。所以以上代码可以修改为：

```cpp
X x(11, 7.0f);
```

另外，前面的章节曾提到过带大括号的列表初始化是不支持缩窄转换的，但是带小括号的列表初始化却是支持缩窄转换的，比如：

```cpp
struct X {
	int i;
	short f;
};

X x1{ 11, 7.0 }; // 编译失败，7.0从double转换到short是缩窄转换
X x2{ 11, 7.0 }; // 编译成功
```

因为扩展的聚合类型改版了原本聚合类型的定义，这就导致了一些**兼容性问题**，这种情况在C++新特性中并不多见。如果不能牢固地掌握新定义的知识点，很容易导致代码无法通过编译，更严重的可能是导致代码运行出现逻辑错误，类似这种Bug又往往难以定位，所以对于扩展的聚合类型我们尤其需要重视起来。

# `override`和`final`说明符（C++11）

## 重写、重载和隐藏

重写（override）、重载（overload）和隐藏（overwrite）在C++中是3个完全不同的概念，但是在平时的工作交流中，我发现有很多C++程序员对它们的概念模糊不清，经常误用或者混用这3个概念，所以在说明`override`说明符之前，我们先梳理一下三者的区别。

- 重写（override）的意思更接近覆盖，在C++中是指派生类覆盖了基类的虚函数，这里的覆盖必须满足有相同的函数签名和返回类型，也就是说有相同的函数名、形参列表以及返回类型。

- 重载（overload），它通常是指在同一个类中有两个或者两个以上函数，它们的函数名相同，但是函数签名不同，也就是说有不同的形参。这种情况在类的构造函数中最容易看到，为了让类更方便使用，我们经常会重载多个构造函数。

- 隐藏（overwrite）的概念也十分容易与上面的概念混淆。隐藏是指基类成员函数，无论它是否为虚函数，当派生类出现同名函数时，如果派生类函数签名不同于基类函数，则基类函数会被隐藏。如果派生类函数签名与基类函数相同，则需要确定基类函数是否为虚函数，如果是虚函数，则这里的概念就是重写；否则基类函数也会被隐藏。另外，如果还想使用基类函数，可以使用`using`关键字将其引入派生类。

## 重写引发的问题

在编码过程中，重写虚函数很容易出现错误，原因是C++语法对重写的要求很高，稍不注意就会无法重写基类虚函数。更糟糕的是，即使我们写错了代码，编译器也可能不会提示任何错误信息，直到程序编译成功后，运行测试才会发现其中的逻辑问题，例如：

```cpp
class Base {
public:
	virtual void some_func() {}
	virtual void foo(int x) {}
	virtual void bar() const {}
	void baz() {}
};

class Derived : public Base {
public:
	virtual void sone_func() {}
	virtual void foo(int &x) {}
	virtual void bar() {}
	virtual void baz() {}
};
```
## 使用override说明符

C++11标准提供了一个非常实用的override说明符，这个说明符必须放到虚函数的尾部，它明确告诉编译器这个虚函数需要覆盖基类的虚函数，一旦编译器发现该虚函数不符合重写规则，就会给出错误提示。

```cpp
class Base {
public:
	virtual void some_func() {}
	virtual void foo(int x) {}
	virtual void bar() const {}
	void baz() {}
};

class Derived : public Base {
public:
	virtual void sone_func override {}
	virtual void foo(int &x) override {}
	virtual void bar() override {}
	virtual void baz() override {}
};
```

## 使用final说明符

在C++中，我们可以为基类声明纯虚函数来迫使派生类继承并且重写这个纯虚函数。但是一直以来，C++标准并没有提供一种方法来阻止派生类去继承基类的虚函数。C++11标准引入`final`说明符解决了上述问题，**它告诉编译器该虚函数不能被派生类重写**。`final`说明符用法和`override`说明符相同，需要声明在虚函数的尾部。

```cpp
class Base {
public:
	virtual void foo(int x) {}
};

class Derived : public Base {
public:
	void foo(int x) final {};
};

class Derived2 : public Derived {
public:
	void foo(int x) {};
};
```

请注意`final`和`override`说明符的一点区别，`final`说明符可以修饰最底层基类的虚函数而`override`则不行，所以在这个例子中`final`可以声明基类`Base`的虚函数`foo`，只不过我们通常不会这样做。

有时候，`override`和`final`会同时出现。这种情况通常是**由中间派生类继承基类后，希望后续其他派生类不能修改本类虚函数的行为而产生的**，举个例子：

```cpp
class Base {
public:
	virtual void log(const char*) const {...}
	virtual void foo(int x) {}
};

class BaseWithFileLog : public Base {
public:
	virtual void log(const char*) const override final {...}
};

class Derived : BaseWithFileLog {
public:
	void foo(int x) {};
};
```

这样一来，后续的派生类`Derived`只能重写虚函数`foo`而无法修改日志函数，保证了日志的一致。

最后要说明的是，`final`说明符不仅能声明虚函数，还可以声明类。**如果在类定义的时候声明了`final`，那么这个类将不能作为基类被其他类继承**，例如：

```cpp
class Base final {
public:
	virtual void foo(int x) {}
};

class Derived : public Base {
public:
	void foo(int x) {};
};
```

在上面的代码中，由于`Base`被声明为`final`，因此`Derived`继承`Base`会在编译时出错。

## `override`和`final`说明符的特别之处

为了和过去的C++代码保持兼容，增加保留的关键字需要十分谨慎。因为一旦增加了某个关键字，过去的代码就可能面临大量的修改。所以在C++11标准中，`override`和`final`并没有被作为保留的关键字，其中`override`只有在虚函数尾部才有意义，而`final`只有在虚函数尾部以及类声明的时候才有意义，因此以下代码仍然可以编译通过：

```cpp
class X {
public:
	void override() {}
	void final() {}
};
```

不过，为了避免不必要的麻烦，建议读者不要将它们作为标识符来使用。

# 基于范围的for循环（C++11 C++17 C++20）

## 烦琐的容器遍历

通常遍历一个容器里的所有元素会用到for循环和迭代器，在大多数情况下我们并不关心迭代器本身:

```cpp
std::map<int, std::string> index_map{ {1, "hello"}, {2, "world"}, {3, "!"}};

std::map<int, std::string>::iterator it = index_map.begin();
for (; it != index_map.end(); it++) {
	std::cout << "key=" << (*it).first << ", value=" << (*it).second << std::endl;
}
```

对于这个问题的一个可行的解决方案是使用标准库提供的`std::for_each`函数，使用该函数只需要提供容器开始和结束的迭代器以及执行函数或者仿函数即可，例如：

```cpp
std::map<int, std::string> index_map{ {1, "hello"}, {2, "world"}, {3, "!"}};

void print(std::map<int, std::string>::const_reference e) {
	std::cout << "key = " << e.first << ", value = " << e.second << std::endl;
}

std::for_each(index_map.begin(), index_map.end(), print);
```

## 基于范围的for循环语法

C++11标准引入了基于范围的for循环特性，该特性隐藏了迭代器的初始化和更新过程，让程序员只需要关心遍历对象本身，其语法也比传统for循环简洁很多

```cpp
for (range_declaration : range_expression) loop_statement
```

范围声明是一个变量的声明，其类型是范围表达式中元素的类型或者元素类型的引用。而范围表达式可以是数组或对象，对象必须满足以下2个条件中的任意一个。

- 对象类型定义了begin和end成员函数

- 定义了以对象类型为参数的begin和end普通函数。

```cpp
#include <iostream>
#include <string>
#include <map>

std::map<int, std::string>index_map{ {1, "hello"}, {2, "world"}, {3, "!"} };
int int_array[] = { 0, 1, 2, 3, 4, 5 };

int main() {
	for (const auto &e : index_map) {
		std::cout << "key = " << e.first << ", value = " << e.second << std::endl;
	}
	for (auto e : int_array) {
		std::cout << e << std::endl;
	}
}
```


```cpp
#include <vector>
struct X {
	X() { std::cout << "default ctor" << std::endl; }
	X(const X& other) {
		std::cout << "copy ctor" << std::endl;
	}
};

int main() {
	std::vector<X> x(10);
	std::cout << "for (auto n: x)" << std::endl;
	for (auto n: x) {
	}
	std::cout << "for (const auto &n : x)" << std::endl;
	for (const auto &n: x) {
	}
}
```

编译运行上面这段代码会发现`for(auto n : x)`的循环调用10次复制构造函数，如果类`X`的数据量比较大且容器里的元素很多，那么这种复制的代价是无法接受的。而`for(const auto &n : x)`则解决了这个问题，整个循环过程没有任何的数据复制。

## begin和end函数不必返回相同类型

在C++11标准中基于范围的for循环相当于以下伪代码

```cpp
{
	auto && __range = range_expression;
	for (auto __begin = begin_expr, __end = end_expr; __begin != __end; ++__begin) {
		range_declaration = *__begin;
		loop_statement
	}
}
```

这段伪代码有一个特点，它要求begin_expr和end_expr返回的必须是同类型的对象。但实际上这种约束完全没有必要，只要`__begin !=__end`能返回一个有效的布尔值即可，所以C++17标准对基于范围的for循环的实现进行了改进，伪代码如下：

```cpp
{
	auto && __range = range_expression;
	auto __begin = begin_expr;
	auto __end = end_expr;
	for (; __begin != __end; ++__begin) {
		range_declaration = *__begin;
		loop_statement
	}
}
```

可以看到，以上伪代码将__begin和__end分离到两条不同的语句，不再要求它们是相同类型。

## 临时范围表达式的陷阱

读者是否注意到了，无论是C++11还是C++17标准，基于范围的for循环伪代码都是由以下这句代码开始的

```cpp 
auto && __range = range_expression;
```

理解了右值引用的读者应该敏锐地发现了这里存在的陷阱`auto&&`。对于这个赋值表达式来说，如果`range_expression`是一个纯右值，那么右值引用会扩展其生命周期，保证其整个`for`循环过程中访问的安全性。但如果`range_ expression`是一个泛左值，那结果可就不确定了，参考以下代码：

```cpp
class T {
	std::vector<int> data_;
public:
	std::vector<int>& items() { return data_; }
	// ...
};

T foo() {
	T t;
	return t;
}

for (auto& x : foo().items()) {} // 未定义行为
```

请注意，这里的`for`循环会引发一个未定义的行为，因为`foo().items()`返回的是一个泛左值类型`std::vector<int>&`，于是右值引用无法扩展其生命周期，导致`for`循环访问无效对象并造成未定义行为。对于这种情况请读者务必小心谨慎，将数据复制出来是一种解决方法：

```cpp
T thing = foo();
for (auto& x : thing.items()) {}
```

在C++20标准中，基于范围的for循环增加了对初始化语句的支持，所以在C++20的环境下我们可以将上面的代码简化为：

```cpp
for (T thing = foo(); auto & x: thing.items()) {}
```

## 实现一个支持基于范围的for循环的类

前面用大量篇幅介绍了使用基于范围的for循环遍历数组和标准容器的方法，实际上我们还可以让自定义类型支持基于范围的for循环。要完成这样的类型必须先实现一个类似标准库中的迭代器。

- 该类型必须有一组和其类型相关的`begin`和`end`函数，它们可以是类型的成员函数，也可以是独立函数。

- `begin`和`end`函数需要返回一组类似迭代器的对象，并且这组对象必须支持`operator *`、`operator !=`和`operator ++`运算符函数

请注意，这里的`operator ++`应该是一个前缀版本，它需要通过声明一个不带形参的`operator ++`运算符函数来完成。下面是一个完整的例子：

```cpp
#include <iostream>
class IntIter {
public:
	IntIter(int* p) : p_(p) {}
	bool operator != (const IntIter& other) {
		return (p_ != other.p_);
	}
	
	const IntIter& operator++() {
		p_++;
		return *this;
	}
	
	int operator*() const {
		return *p_;
	}
	
private:
	int *p_;
};

template<unsigned int fix_size>
class FixIntVector {
public:
	FixIntVector(std::initializer_list<int> init_list) {
		int *cur = data_;
		for (auto e : init_list) {
			*cur = e;
			cur++;
		}
	}
	
	IntIter begin() {
		return IntIter(data_);
	}
	
	IntIter end() {
		return IntIter(data_ + fix_size);
	}
private:
	int data_[fix_size]{0};
};

int main() {
	FixIntVector<10> fix_int_vector {1, 3, 5, 7, 9};
	for (auto e: fix_int_vector) {
		std::cout << e << std::endl;
	}
}
```

请注意，这里使用成员函数的方式实现了begin和end，但有时候需要遍历的容器可能是第三方提供的代码。这种情况下我们可以实现一组独立版本的begin和end函数，**这样做的优点是能在不修改第三方代码的情况下支持基于范围的for循环。**


# 支持初始化语句的 `if` 和 `switch`（C++17）

在C++17标准中，if控制结构可以在执行条件语句之前先执行一个初始化语句。语法如下：

```cpp
if (init; condition) {}
```

其中`init`是初始化语句，`condition`是条件语句，它们之间使用分号分隔。允许初始化语句的`if`结构让以下代码成为可能：

```cpp
#include <iostream>
bool foo() {
	return true;
}

int main() {
	if (bool b = foo(); b) {
		std::cout << std::boolalpha << "good! foo()=" << b << std::endl;
	}
}
```

在初始化语句中声明的变量b能够在if的作用域继续使用。事实上，该变量的生命周期会一直伴随整个if结构，包括else if和else部分。

if初始化语句中声明的变量拥有和整个if结构一样长的声明周期，所以前面的代码可以等价于：

```cpp
#include <iostream>
bool foo() {
	return true;
}

int main() {
	{
		bool b = foo();
		if (b) {
			std::cout << std::boolalpha << "good! foo()=" << b << std::endl;
		}
	}
}
```

当然，我们还可以在if结构中添加else部分：

```cpp
if (bool b = foo(); b) {
	std::cout << std::boolalpha << "good! foo() = " << b << std::endl;
} else {
	std::cout << std::boolalpha << "bad! foo()=" << b << std::endl;
}
```

在`if`结构中引入`else if`后，情况会稍微变得复杂一点，因为在`else if`条件语句之前也可以使用初始化语句：

```cpp
#include <iostream>
bool foo() {
	return false;
}

bool bar() {
	return true;
}

int main() {
	if (bool b = foo(); b) {
		std::cout << std::boolalpha << "foo() = " << b << std::endl;
	} else if (bool b1 = bar(); b1) {
		std::cout << std::boolalpha << "foo() = " << b << ", bar() = " << b1 << std::endl;
	}
}
```

以上的代码等价于

```cpp
{
	bool b = foo();
	if (b) {
		std::cout << std::boolalpha << "foo() = " << b << std::endl;
	} else {
		bool b1 = bar();
		if (b1) {
			std::cout << std::boolalpha << "foo() = " << b << ", bar() = " << b1 << std::endl;
		}
	}
}
```

因为`if`初始化语句声明的变量会贯穿整个`if`结构，所以我们可以利用该特性对整个if结构加锁，例如：

```cpp
#include <mutex>
std::mutex mx;
bool  shared_flag = true;
int main() {
	if (std::lock_guard<std::mutex> lock(mx); shared_flag) {
		shared_flag = false;
	}
}
```

继续扩展思路，从本质上来说初始化语句就是在执行条件判断之前先执行了一个语句，并且语句中声明的变量将拥有与`if`结构相同的生命周期。所以我们在代码中没有必要一定在初始化语句中初始化判断条件的变量，如`if(std::lock_guard <std::mutex>lock(mx); shared_flag)`，初始化语句并没有初始化条件判断的变量`shared_flag`。类似的例子还有：

```cpp
#include <cstdio>
#include <string>
int main() {
	std::string str;
	if (char buf[10]{0}; std::fgets(buf, 10, stdin)) {
		str += buf;
	}
}
```

## 支持初始化语句的switch

这里以`std::condition_variable`为例，其成员函数`wait_for`需要一个`std:: unique_lock<std::mutex>&`类型的实参，于是在`switch`的初始化语句中可以构造一个`std::unique_lock<std::mutex>`类型的对象，具体代码如下：

```cpp
#include <condition_variable>
#include <chrono>

using namespace std::chrono_literals;
std::condition_variable cv;
std::mutex cv_m;
int main() {
	switch (std::unique_lock<std::mutex> lk(cv_m); cv.wait_for(lk, 100ms)) {
		case std::cv_status::timeout: break;
		case std::cv_status::no_timeout: break;
	}
}
```

`switch`初始化语句声明的变量的生命周期会贯穿整个`switch`结构，这一点和`if`也相同，所以变量`lk`能够引用到任何一个`case`的分支中。

# `static_assert` 声明

## 运行时断言

在静态断言出现以前，我们使用的是运行时断言，只有程序运行起来之后才有可能触发它。通常情况下运行时断言只会在Debug模式下使用，因为断言的行为比较粗暴，它会直接显示错误信息并终止程序。

还有一点需要注意，断言不能代替程序中的错误检查，它只应该出现在需要表达式返回true的位置,相反，如果表达式中涉及外部输入，则不应该依赖断言

```cpp
void resize_buffer(void* buffer, int new_size) {
	assert(buffer != nullptr); // OK，用assert检查函数参数
	assert(new_size > 0);
	assert(new_size <= MAX_BUFFER_SIZE);
	...
}

bool get_user_input(char c) {
	assert(c == '\0x0d'); // 不合适，assert不应该用于检查外部输入
	...
}
```

## 静态断言的需求

虽然运行时断言可以满足一部分需求，但是它有一个缺点就是必须让程序运行到断言代码的位置才会触发断言。如果想在模板实例化的时候对模板实参进行约束，这种断言是无法办到的。我们需要一个能在编译阶段就给出断言的方法。可惜在C++11标准之前，没有一个标准方法来达到这个目的，我们需要利用其他特性来模拟。下面给出几个可行的方案：

```cpp
#define STATIC_ASSERT_CONCAT_IMP(x, y) x ## y
#define STATIC_ASSERT_CONCAT(x, y) STATIC_ASSERT_CONCAT_IMP(x, y)

// 方案1
#define STATIC_ASSERT(expr)  \ 
	do {                     \	
			char STATIC_ASSERT_CONCAT(   \
			static_assert_var, __COUNTER__)   \
			[(expr) != 0 ? 1 : -1];           \
	} while (0)

template<bool>
struct static_assert_st;
template<>
struct static_assert_at<true> {};

// 方案2
#define STATIC_ASSERT2(expr)  \
	static_assert_st<(expr) != 0>()

// 方案3
#define STATIC_ASSERT3(expr)  \ 
	static_assert_st<(expr) != 0> \
	STATIC_ASSERT_CONCAT(  \
	static_assert_var, __COUNTER__)
```

以上代码的方案1，利用的技巧是数组的大小不能为负值，当expr表达式返回结果为false的时候，条件表达式求值为−1，这样就导致数组大小为−1，自然就会引发编译失败。方案2和方案3则是利用了C++模板特化的特性，当模板实参为true的时候，编译器能找到特化版本的定义。但当模板参数为false的时候，编译器无法找到相应的特化定义，从而编译失败。方案2和方案3的区别在于，方案2会构造临时对象，这让它无法出现在类和结构体的定义当中。而方案3则声明了一个变量，可以出现在结构体和类的定义中，但是它最大的问题是会改变结构体和类的内存布局。总而言之，虽然我们可以在一定程度上模拟静态断言，但是这些方案并不完美。

## 静态断言

`static_assert`声明是C++11标准引入的特性，用于在程序编译阶段评估常量表达式并对返回false的表达式断言，我们称这种断言为静态断言。它基本上满足我们对静态断言的要求。

- 所有处理必须在编译期间执行，不允许有空间或时间上的运行时成本。

- 它必须具有简单的语法。

- 断言失败可以显示丰富的错误诊断信息。

- 它可以在命名空间、类或代码块内使用。

- 失败的断言会在编译阶段报错。

C++11标准规定，使用`static_assert`需要传入两个实参：常量表达式和诊断消息字符串。请注意，第一个实参必须是常量表达式，因为编译器无法计算运行时才能确定结果的表达式：

```cpp
#include <type_traits>

class A {
};

class B : public A {
};

class C {
};

template<class T>
class E {
	static_assert(std::is_base_of<A, T>::value, "T is not base of A");
};

int main(int argc, char* argv[]) {
	static_assert(argc > 0, "argc > 0");  // 使用错误，argc>0不是常量表达式
	E<C> x;   // 使用正确，但由于A不是C的基类，所以触发断言
	static_assert(sizeof(int) >= 4);      , // 使用正确，表达式返回真，不会触发失败断言
	E<B> y;  // 使用正确，A是B的基类，不会触发失败断言
}
```


## 单参数static_assert

不知道读者是否和我有同样的想法，在大多数情况下使用`static_assert`的时候输入的诊断信息字符串就是常量表达式本身，所以让常量表达式作为诊断信息字符串参数的默认值是非常理想的。为了达到这个目的，我们可以定义一个宏：

```cpp
#define LAZY_STATIC_ASSERT(B) static_assert(B, #B)
```

在支持C++17标准的环境中，我们可以忽略第二个参数：

```cpp
#include <type_traits>

class A{
};

class B : public A {
};

class C {
};

template<class T>
class E {
	static_assert(std::is_base_of(A, T)::value);
};

int main(int argc, char* argv[]) {
	E<C> x;  // 使用正确，但由于A不是C的基类，会触发失败断言
	static_assert(sizeof(int) < 4);  // 使用正确，但表达式返回false，会触发失败断言
}
```

不过在GCC上，即使指定使用C++11标准，GCC依然支持单参数的`static_assert`。MSVC则不同，要使用单参数的`static_assert`需要指定C++17标准。

# 结构化绑定（C++17 C++20）

## 使用结构化绑定

熟悉Python的读者应该知道，Python函数可以有多个返回值，例如：

```cpp
def return_multiple_values():
	return 11, 7
	
x, y = return_multiple_values()
```

在C++11标准中同样引入了元组的概念，通过元组C++也能返回多个值

```cpp
#include <iostream>
#include <tuple>

std::tuple<int, int> return_multiple_values() {
	return std::make_tuple(11, 7);
}

int main() {
	int x = 0, y = 0;
	std::tie(x, y) = return_multiple_values();
	std::cout << "x = " << x << " y = " << y < std::endl;
}
```

可以看到，这段代码和Python完成了同样的工作，但代码却要麻烦许多。其中一个原因是C++11必须指定`return_multiple_values`函数的返回值类型，另外，在调用`return_multiple_values`函数前还需要声明变量`x`和`y`，并且使用函数模板`std::tie`将`x`和`y`通过引用绑定到`std::tuple<int&,int&>`上。对于第一个问题，我们可以使用C++14中`auto`的新特性来简化返回类型的声明（可以回顾第3章）：

```cpp
auto return_multiple_values() {
	return std::make_tuple(11, 7);
}
```

要想解决第二个问题就必须使用C++17标准中新引入的特性——结构化绑定。所谓结构化绑定是指将一个或者多个名称绑定到初始化对象中的一个或者多个子对象（或者元素）上，相当于给初始化对象的子对象（或者元素）起了别名，请注意别名不同于引用，这一点会在后面详细介绍。

```cpp
#include <iostream>
#include <tuple>

auto return_multiple_values() {
	return std::make_tuple(11, 7);
}

int main() {
	auto[x, y] = return_multiple_values();
	std::cout << " x = " << x << " y = " << y << endl;
}
```

在上面这段代码中，`auto[x, y] =return_multiple_values()`是一个典型的结构化绑定声明，其中`auto`是类型占位符，`[x, y]`是绑定标识符列表，其中x和y是用于绑定的名称，绑定的目标是函数`return_multiple_values()`返回结果副本的子对象或者元素。

请注意，结构化绑定的目标不必是一个函数的返回结果，实际上等号的右边可以是任意一个合理的表达式，比如：

```cpp
#include <iostream>
#include <string>

struct BindTest {
	int a = 42;
	std::string b = "hello structured binding";
};

int main() {
	BindTest bt;
	auto[x, y] = bt;
	std::cout << " x = " << x << " y = " << y << std::endl;
}
```

可以看到结构化绑定能够直接绑定到结构体上。将其运用到基于范围的for循环中会有更好的效果：

```cpp
#include <iostream>
#include <string>
#include <vector>

struct BindTest {
	int a = 42;
	std::string b = "hello structured binding";
};

int main() {
	std::vector<BindTest> bt{ {11, "hello"}, {7, "c++"}, {42, "world"}};
	for (const auto& [x, y] : bt) {
		std::cout << "x = " << x << " y = " << y << std::endl;
	}
}
```

在这个基于范围的for循环中，通过结构化绑定直接将x和y绑定到向量bt中的结构体子对象上，省去了通过向量的元素访问成员变量a和b的步骤。

## 深入理解结构化绑定

在阅读了前面的内容之后，读者是否有这样的理解

1. 结构化绑定的目标就是等号右边的对象。

2. 所谓的别名就是对等号右边对象的子对象或者元素的引用。

上面的理解是错误的。**真实的情况是，在结构化绑定中编译器会根据限定符生成一个等号右边对象的匿名副本，而绑定的对象正是这个副本而非原对象本身。** 另外，这里的别名真的是单纯的别名，别名的类型和绑定目标对象的子对象类型相同，而引用类型本身就是一种和非引用类型不同的类型。在初步了解了结构和绑定的“真相”之后，现在我将使用伪代码进一步说明它是如何工作起来的。对于结构化绑定代码：

```cpp
BindTest bt;
const auto [x, y] = bt;
```

编译器为其生成的代码大概是这样的：

```cpp
BindTest bt;
const auto _anonymous = bt;
aliasname x = _anonymous.a
aliasname y = _anonymous.b
```

在上面的伪代码中，`_anonymous`是编译器生成的匿名对象，可以注意到`const auto [x, y] = bt`中`auto`的限定符会直接应用到匿名对象`_anonymous`上。也就是说，`_anonymous`是`const`还是`volatile`完全依赖`auto`的限定符。另外，在伪代码中`x`和`y`的声明用了一个不存在的关键字`aliasname`来表达它们不是`_anonymous`成员的引用而是`_anonymous`成员的别名，也就是说`x`和`y`的类型分别为`const int`和`const std:: string`，而不是`const int&`和`const std::string&`。为了证明以上两点，读者可以尝试编译运行下面这段代码：

```cpp
#include <iostream>
#include <string>

struct BindTest {
	int a = 42;
	std::string b = "hello structured binding";
};

int main() {
	BindTest bt;
	const auto[x, y] = bt;
	std::cout << "&bt.a = " << &bt.a << " &x = " << &x << std::endl;
	std::cout << "&bt.b = " << &bt.b << " &y = " << &y << std::endl;
	std::cout << "std::is_same_v<const int, decltype(x)> = " << std::is_same_v<const int, decltype(x)> << std::endl;
	std::cout << "std::is_same_v<const std::string,decltype(y)> = " << std::is_same_v<const std::string, decltype(y)> << std::endl; 
}
```

编译运行的结果如下：

```txt
&bt.a=0x77fde0 &x=0x77fd80
&bt.b=0x77fde8 &y=0x77fd88
std::is_same_v<const int, decltype(x)>=1
std::is_same_v<const std::string, decltype(y)>=1
```

正如上文中描述的那样，别名`x`并不是`bt.a`，因为它们的内存地址不同。另外，`x`和`y`的类型分别与`const int`和`conststd::string`相同也证明了它们是别名而不是引用的事实。由此可见，如果在上面这段代码中试图使用`x`和`y`去修改`bt`的数据成员是无法成功的，因为一方面`x`和`y`都是常量类型；另一方面即使`x`和`y`是非常量类型，改变的`x`和`y`只会影响匿名对象而非`bt`本身。当然了，了解了结构化绑定的原理之后，写一个能改变`bt`成员变量的结构化绑定代码就很简单了：

```cpp
int main() {
	BindTest bt;
	auto& [x, y] = bt;
	std::cout << "&bt.a = " << &bt.a << " &x = " << &x << std::endl;
	std::cout << "&bt.b = " << &bt.b << " &y = " << &y << std::endl;
	x = 11;
	std::cout << "bt.a = " << bt.a << std::endl;
	bt.b = "hi structured binding";
	std::cout << "y = " << y << std::endl;
}
```

虽然只是将`const auto`修改为`auto&`，但是已经能达到让`bt`数据成员和`x`、`y`相互修改的目的了：

```txt
BindTest bt;
auto &_anonymous = bt;
aliasname x = _anonymous.a
aliasname y = _anonymous.b
```

关于引用有趣的一点是，如果结构化绑定声明为`const auto& [x, y] = bt`，那么`x = 11`会编译失败，因为`x`绑定的对象是一个常量引用，而`bt.b = "hi structured binding"`却能成功修改`y`的值，因为`bt`本身不存在常量问题。

请注意，使用结构化绑定无法忽略对象的子对象或者元素：

```cpp
auto t = std::make_tuple(42, "hello world");
auto [x] = t;
```

以上代码是无法通过编译的，必须有两个别名分别对应`bt`的成员变量`a`和`b`。熟悉C++11的读者可能会提出仿照`std::tie`使用`std::ignore`的方案：

```cpp
auto t = std::make_tuple(42, "hello world");
int x = 0, y = 0;
std::tie(x, std::ignore) = t;
std::tie(y, std::ignore) = t;
```

虽然这个方案对于`std::tie`是有效的，但是结构化绑定的别名还有一个限制：无法在同一个作用域中重复使用。这一点和变量声明是一样的，比如：

```cpp
auto t = std::make_tuple(42, "hello world");
auto[x, ignore] = t;
auto[y, ignore] = t; // 编译错误，ignore无法重复声明
```

## 结构化绑定的3种类型

结构化绑定可以作用于3种类型，包括原生数组、结构体和类对象、元组和类元组的对象，接下来将一一介绍。

### 绑定到原生数组

绑定到原生数组即将标识符列表中的别名一一绑定到原生数组对应的元素上。所需条件仅仅是要求别名的数量与数组元素的个数一致，比如：

```cpp
#include <iostream>

int main() {
	int a[3]{ 1, 3, 5 };
	auto[x, y, z] = a;
	std::cout << "[x, y, z] = [" << x << ", " << y << ", " << z << "]" << endl;
}
```

另外，绑定到原生数组需要小心数组的退化，因为在绑定的过程中编译器必须知道原生数组的元素个数，一旦数组退化为指针，就将失去这个属性。

### 绑定到结构体和类对象

首先，**类或者结构体中的非静态数据成员个数必须和标识符列表中的别名的个数相同**；其次，这些**数据成员必须是公有的**（C++20标准修改了此项规则，详情见20.5节）；这些数据成员必须是在同一个类或者基类中；最后，**绑定的类和结构体中不能存在匿名联合体**：

```cpp
class BindTest {
	int a = 42;   // 私有成员变量
public:
	double b = 11.7;
};

int main() {
	BindTest bt;
	auto[x, y] = bt;
}
```

以上代码会编译错误，因为BindTest成员变量a是私有的，违反了绑定结构体的限制条件：

```cpp
class BindBase1 {
public:
	int a = 42;
	double b = 11.7;
};

class BindTest1 : public BindBase1 {};

class BindBase2 {};

class BindTest2 : public BindBase2 {
public:
	int a = 42;
	double b = 11.7;
};

class BindBase3 {
public:
	int a = 42;
};

class BindTest3 : public BindBase3 {
public:
	double b = 11.7;
};

int main() {
	BindTest1 bt1;
	BindTest2 bt2;
	BindTest3 bt3;
	auto[x1, y1] = bt1;  // 编译成功
	auto[x2, y2] = bt2;  // 编译成功
	auto[x3, y3] = bt3;  // 编译错误
}
```

### 绑定到元组和类元组的对象

绑定元组和类元组有一系列抽象的条件：对于元组或者类元组类型T。

- 需要满足`std::tuple_size<T>::value`是一个符合语法的表达式，并且该表达式获得的整数值与标识符列表中的别名个数相同。

- 类型T还需要保证`std::tuple_element<i, T>::type`也是一个符合语法的表达式，其中`i`是小于`std::tuple_size<T>::value`的整数，表达式代表了类型`T`中第`i`个元素的类型。

- 类型T必须存在合法的成员函数模板`get<i>()`或者函数模板`get<i>(t)`，其中i是小于`std::tuple_size<T>::value`的整数，`t`是类型`T`的实例，`get<i>()`和`get<i>(t)`返回的是实例`t`中第`i`个元素的值。

获取这些条件特征的代价也并不高，只需要为目标类型提供`std::tuple_size`、`std::tuple_element`以及`get`的特化或者偏特化版本即可。实际上，标准库中除了元组本身毫无疑问地能够作为绑定目标以外，`std::pair`和`std::array`也能作为结构化绑定的目标，其原因就是它们是满足上述条件的类元组。说到这里，就不得不进一步讨论`std::pair`了，因为它对结构化绑定的支持给我们带来了一个不错的惊喜：

```cpp
#include <iostream
#include <string>
#include <map>

int main() {
	std::map<int, std::string> id2str{ {1, "hello"}, {3, "Structured"}, {5, "bindigs"}};
	for (const auto& elem: id2str) {
		std::cout << "id = " << elem.first << ", std = " << elem.second << std::endl;
	}
}
```

加入结构化绑定后代码将被进一步简化。我们可以将std::pair的成员变量first和second绑定到别名以保证代码阅读起来更加清晰：

```cpp
for (const auto&[id, std]:id2str) {
	std::cout << "id = " << id << ", str = " << str << std::endl;
}
```

## 实现一个类元组类型

以上一节中提到的`BindTest3`为例，我们知道由于它的数据成员分散在派生类和基类之中，因此无法使用结构化绑定。下面将通过让其满足类元组的条件，从而达到支持结构化绑定的目的：

```cpp
#include <iostream>
#include <tuple>

class BindBase3 {
public:
	int a = 42;
};

class BindTest3 : public BindBase3 {
public:
	double b = 11.7;
};

namespace std {
	template<>
	struct tuple_size<BindTest3> {
		static constexpr size_t value = 2;
	};
	
	template<>
	struct tuple_element<0, BindTest3> {
		using type = int;
	};
	
	template<>
	struct tuple_element<1, BindTest3> {
		using type = double;
	};
}

template<std::size_t Idx>
auto& get(BindTest3 &bt) = delete;

template<>
auto& get<0>(BindTest3 &bt) { return bt.a; }

template<>
auto& get<1>(BindTest3 &bt) { return bt.b; }

int main() {
	BindTest3 bt3;
	auto& [x3, y3] = bt3;
	x3 = 78;
	std::cout << bt3.a << std::endl;
}
```

在上面这段代码中，我们为BindTest3实现了3种特性以满足类元组的限制条件。首先实现的是：

```cpp
template<>
struct tuple_size<BindTest3> {
	static constexpr size_t value = 2;
};
```

它的作用是告诉编译器将要绑定的子对象和元素的个数，这里通过特化让`tuple_size<BindTest3>::value`的值为2，也就是存在两个子对象。然后需要明确的是每个子对象和元素的类型：

```cpp
template<>
struct tuple_element<0, BindTest3> {
	using type = int;
};

template<>
struct tuple_element<1, BindTest3> {
	using type = double;
};
```

最后`template<std::size_t Idx> auto& get(BindTest3 &bt) = delete`可以明确地告知编译器不要生成除了特化版本以外的函数实例以防止`get`函数模板被滥用。

正如上文强调的，我不推荐实现成员函数版本的get函数，**因为这需要修改原有的代码**。但是当我们重新编写一个类，并且希望它支持结构化绑定的时候，也不妨尝试实现几个get成员函数：

```cpp
#include <iostream>
#include <tuple>

class BindBase3 {
public:
	int a = 42;
};

class BindTest3 : public BindBase3 {
public:
	double b = 11.7;
	template<std::size_t Idx> auto& get() = delete;
};

template<> auto& BindTest3::get<0> { return a; }
template<> auto& BindTest3::get<1> { return b; }

namespace std {
	template<>
	struct tuple_size<BindTest3> {
		static constexpr size_t value = 2;
	};
	
	template<>
	struct tuple_element<0, BindTest3> {
		using type = int;
	};
	
	template<>
	struct tuple_element<1, BindTest3> {
		using type = double;
	};
}

int main() {
	BindTest3 bt3;
	auto& [x3, y3] = bt3;
	x3 = 78;
	std::cout << bt3.a << std::endl;
}
```

这段代码和第一份实现代码基本相同，我们只需要把精力集中到get成员函数的部分：

```cpp
class BindTest3 : public BindBase3 {
public:
	double b = 11.7;
	template<std::size_t Idx> auto& get() = delete;
};

template<> auto& BindTest3::get<0>() { return a; }
template<> auto& BindTest3::get<1>() { return b; }
```

这段代码中`get`成员函数的优势显而易见，成员函数不需要传递任何参数。另外，特化版本的函数`get<0>`和`get<1>`可以直接返回`a`和`b`，这显得格外简洁。读者不妨自己编译运行一下这两段代码，其输出结果应该都是78，修改`bt.a`成功。

## 绑定的访问权限问题

当在结构体或者类中使用结构化绑定的时候，需要有公开的访问权限，否则会导致编译失败。这条限制乍看是合理的，但是仔细想来却引入了一个相同条件下代码表现不一致的问题：

```cpp
struct A {
	friend void foo();
private:
	int i;
};

void foo() {
	A a{};
	auto x = a.i;  // 编译成功
	auto [y] = a;  // 编译失败
}
```

在上面这段代码中，`foo`是结构体`A`的友元函数，它可以访问`A`的私有成员`i`。但是，结构化绑定却失败了，这就明显不合理了。同样的问题还有访问自身成员的时候：

```cpp
class C {
	int i;
	void foo(const C& other) {
		auto [x] = other; // 编译失败
	}
}
```

为了解决这类问题，C++20标准规定结构化绑定的限制不再强调必须为公开数据成员，编译器会根据当前操作的上下文来判断是否允许结构化绑定。幸运的是，虽然标准是2018年提出修改的，但在我实验的3种编译器上，无论是C++17还是C++20标准，以上代码都可以顺利地通过编译。

