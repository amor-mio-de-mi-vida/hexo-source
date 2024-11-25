---
date: 2024-11-24 22:58:54
date modified: 2024-11-25 19:38:40
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