---
date: 2024-11-21 20:22:55
date modified: 2024-11-25 11:07:46
title: Modern cpp feature 1~10
tags:
  - cpp
categories:
  - cpp
---
Reference: 《现代C++语言核心特性解析》

1. 新基础类型

2. 内联和嵌套命名空间

3. `auto` 占位符

4. `decltype` 说明符

5. 函数返回类型后置

6. 右值引用

7. lambda 表达式

8. 非静态数据成员默认初始化

9. 列表初始化

10. 默认和删除函数

<!--more-->

# 新基础类型

## 整数类型 `long long`

我们知道long通常表示一个32位整型，而long long则是用来表示一个64位的整型。C++标准还为其定义LL和ULL作为这两种类型的字面量后缀，所以在初始化long long类型变量的时候可以这么写：

```cpp
long long x = 65536LL;
```

当然，这里可以忽略LL这个字面量后缀，直接写成下面的形式也可以达到同样的效果：

```cpp
long long x = 65536;
```

要强调的是，字面量后缀并不是没有意义的，在某些场合下我们必须用到它才能让代码的逻辑正确，比如下面的代码：

```cpp
long long x1 = 65536 << 16; // 计算得到的 x1 值为0
std::cout << "x1 = " << x1 << std::endl;

long long x2 = 65536LL << 16; //计算得到的x2值为 4294967296（0x100000000）
std::cout << "x2 = " << x2 << std::endl;
```

和其他整型一样，long long也能运用于枚举类型和位域，例如：

```cpp
enum lonnglong _num : long long {
	x1, 
	x2
};

struct longlong_struct {
	long long x1 : 8;
	long long x2 : 24;
	long long x3 : 32;
};

std::cout << sizeof(longlong_enum::x1) << std::endl; // 输出大小为8
std::cout << sizeof(longlong_struct) << std::endl // 输出大小为8
```

作为一个新的整型long long，C++标准必须为它配套地加入整型的大小限制。在头文件中增加了以下宏，分别代表long long的最大值和最小值以及unsigned long long的最大值：

```cpp
#define LLONG_MAX 9223372036854775807LL // long long的最大值
#define LLONG_MIN (-9223372036854775807LL - 1) // long long的最小值
#define ULLONG_MAX 0xffffffffffffffffULL // unsigned long long的最大值
```

C++标准中对标准库头文件做了扩展，特化了long long和 unsigned long long版本的numeric_ limits类模板。这使我们能够更便捷地获取这些类型的最大值和最小值

```cpp
#include <iostream>
#include <limits>
#include <cstdio>
int main(int argc, char* argv[]) {
	// 使用宏方法
	std::cout << "LLONG_MAX = " << LLONG_MAX << std::endl;
	std::cout << "LLONG_MIN = " << LLONG_MIN << std::endl;
	std::cout << "ULLONG_MAX = " << ULLONG_MAX << std::endl;
	
	// 使用类模版方法
	std::cout << "std::numeric_limits<long long>::max() = " << std::numeric_limits<long long>::max() << std::endl;
	std::cout << "std::numeric_limits<long long>::min() = " << std::numeric_limits<long long>::min() << std::endl;
	std::cout << "std::numeric_limits<unsigned long long>::max() = " << std::numeric_limits<unsigned long long>::max() << std::endl;
	// 使用 printf 打印输出
	std::printf("LLONG_MAX = %lld\n", LLONG_MAX);
	std::printf("LLONG_MIN = %lld\n", LLONG_MIN);
	std::printf("ULLONG_MAX = %ullu\n", ULLONG_MAX);
}
```

## 新字符类型`char16_t` 和 `char32_t`

在C++11标准中添加两种新的字符类型char16_t和char32_t，它们分别用来对应Unicode字符集的UTF-16和UTF-32两种编码方法。

UTF-8、UTF-16和UTF-32简单来说是使用不同大小内存空间的编码方法。

UTF-32是最简单的编码方法，该方法用一个32位的内存空间（也就是4字节）存储一个字符编码，由于Unicode字符集的最大个数为0x10FFFF（ISO 10646），因此4字节的空间完全能够容纳任何一个字符编码。UTF-32编码方法的优点显而易见，它非常简单，计算字符串长度和查找字符都很方便；缺点也很明显，太占用内存空间。

UTF-16编码方法所需的内存空间从32位缩小到16位（占用2字节），但是由于存储空间的缩小，因此UTF-16最多只能支持0xFFFF个字符，这显然不太够用，于是UTF-16采用了一种特殊的方法来表达无法表示的字符。简单来说，从0x0000～0xD7FF以及0xE000～0xFFFF直接映射到Unicode字符集，而剩下的0xD800～0xDFFF则用于映射0x10000～0x10FFFF的Unicode字符集，映射方法为：字符编码减去0x10000后剩下的20比特位分为高位和低位，高10位的映射范围为0xD800～0xDBFF，低10位的映射范围为0xDC00～0xDFFF。例如0x10437，减去0x10000后的高低位分别为0x1和0x37，分别加上0xD800和0xDC00的结果是0xD801和0xDC37。

幸运的是，一般情况下0xFFFF足以覆盖日常字符需求，我们也不必为了UTF-16的特殊编码方法而烦恼。UTF-16编码的优势是可以用固定长度的编码表达常用的字符，所以计算字符长度和查找字符也比较方便。另外，在内存空间使用上也比UTF-32好得多。

最后说一下我们最常用的UTF-8编码方法，它是一种可变长度的编码方法。由于UTF-8编码方法只占用8比特位（1字节），因此要表达完数量高达0x10FFFF的字符集，它采用了一种前缀编码的方法。这个方法可以用1～4字节表示字符个数为0x10FFFF的Unicode（ISO 10646）字符集。为了尽量节约空间，常用的字符通常用1～2字节就能表达，其他的字符才会用到3～4字节，所以在内存空间可以使用UTF-8，但是计算字符串长度和查找字符在UTF-8中却是一个令人头痛的问题。

### 使用新字符类型 `char16_t` 和 `char32_t`

对于UTF-8编码方法而言，普通类型似乎是无法满足需求的，毕竟普通类型无法表达变长的内存空间。所以一般情况下我们直接使用基本类型char进行处理，而过去也没有一个针对UTF-16和UTF-32的字符类型。到了C++11，char16_t和char32_t的出现打破了这个尴尬的局面。除此之外，C++11标准还为3种编码提供了新前缀用于声明3种编码字符和字符串的字面量，它们分别是UTF-8的前缀u8、UTF-16的前缀u和UTF-32的前缀U：

```cpp
char utf8c = u8'a';
// char utf8c = u8'好';
char16_t utf16c = u'好';
char32_t utf32c = U'好';
char utf8[] = u8"你好世界";
char16_t utf16[] = u"你好世界";
char32_t utf32[] = U"你好世界";
```

### `wchr_t` 存在的问题

在C++98的标准中提供了一个`wchar_t`字符类型，并且还提供了前缀L，用它表示一个宽字符。事实上Windows系统的API使用的就是`wchar_t`，它在Windows内核中是一个最基础的字符类型。`wchar_t`确实在一定程度上能够满足我们对于字符表达的需求，但是起初在定义`wchar_t`时并没有规定其占用内存的大小。于是就给了实现者充分的自由，以至于在Windows上`wchar_t`是一个16位长度的类型（2字节），而在Linux和macOS上`wchar_t`却是32位的（4字节）。这导致了一个严重的后果，我们写出的代码无法在不同平台上保持相同行为。而`char16_t`和`char32_t`的出现解决了这个问题，它们明确规定了其所占内存空间的大小，让代码在任何平台上都能够有一致的表现。

### 新字符串连接

如果两个字符串字面量具有相同的前缀，则生成的连接字符串字面量也具有该前缀。如果其中一个字符串字面量没有前缀，则将其视为与另一个字符串字面量具有相同前缀的字符串字面量，其他的连接行为由具体实现者定义。

### 库对新字符类型的支持

C11增加了4个字符的转换函数，包括：

```cpp
size_t mbrtoc16( char16_t* pc16, const char* s, size_t n, mbstate_t* ps );

size_t c16rtomb( char* s, char16_t c16, mbstate_t* ps );

size_t mbrtoc32( char32_t* pc32, const char* s, size_t n, mbstate_t* ps );

size_t c32rtomb( char* s, char32_t c32, mbstate_t* ps );
```

除此之外，C++标准库的字符串也加入了对新字符类型的支持，例如：

```cpp
using u16string = basic_string;
using u32string = basic_string;
using wstring = basic_string;
```

### `char8_t` 字符类型

使用`char`类型来处理UTF-8字符虽然可行，但是也会带来一些困扰，比如当库函数需要同时处理多种字符时必须采用不同的函数名称以区分普通字符和UTF-8字符。C++20标准新引入的类型`char8_t`可以解决以上问题，它可以代替`char`作为UTF-8的字符类型。`char8_t`具有和`unsigned char`相同的符号属性、存储大小、对齐方式以及整数转换等级。引入`char8_t`类型后，在C++17环境下可以编译的UTF-8字符相关的代码会出现问题，例如：

```cpp
char std[] = u8"text"; // c++ 编译成功; c++20编译失败,需要char8_t
char c = u8'c';
char8_t c8a[] = "text"; // C++20编译失败，需要char
char8_t c8 = 'c';
```

另外，为了匹配新的char8_t字符类型，库函数也有相应的增加：

```cpp
size_t mbrtoc8(char8_t* pc8, const char* s, size_t n, mbstate_t* ps);

size_t c8rtomb(char* s, char8_t c8, mbstate_t* ps);

using u8string = basic_string;
```


# 内联和嵌套命名空间

## 内联命名空间的定义和使用

开发一个大型工程必然会有很多开发人员的参与，也会引入很多第三方库，这导致程序中偶尔会碰到同名函数和类型，造成编译冲突的问题。为了缓解该问题对开发的影响，我们需要合理使用命名空间。程序员可以将函数和类型纳入命名空间中，这样在不同命名空间的函数和类型就不会产生冲突，当要使用它们的时候只需打开其指定的命名空间即可，例如:

```cpp
namespace S1 {
	void foo() {}
}

namespace S2 {
	void foo() {}
}

using namespace S1;

int main() {
	foo;
	S2::foo();
}
```

C++11标准增强了命名空间的特性，提出了内联命名空间的概念。 内联命名空间能够把空间内函数和类型导出到父命名空间中，这样即使不指定子命名空间也可以使用其空间内的函数和类型了，比如：

```cpp
#include <iostream>

namespace Parent {
	namespace Child1 {
		void foo() { std::cout << "Child1::foo()" << std::endl; }
	}
	inline namespace Child2 {
		void foo() { std::out << "Child2::foo()" << std::endl; }
	}
}

int main() {
	Parent::Child1::foo();
	Parent::foo();
}
```

该特性可以帮助库作者无缝升级库代码，让客户不用修改任何代码也能够自由选择新老库代码。举个例子:

```cpp
#include <iostream>

namespace Parent {
	void foo() { std::cout << "foo v1.0" << std::endl; }
}

int main() {
	Parent::foo();
}
```

请注意，示例代码中只能有一个内联命名空间，否则编译时会造 成二义性问题，编译器不知道使用哪个内联命名空间的foo函数。

## 嵌套命名空间的简化语法

有时候打开一个嵌套命名空间可能只是为了向前声明某个类或者函数，但是却需要编写冗长的嵌套代码，加入一些无谓的缩进，这很难让人接受。幸运的是，C++17标准允许使用一种更简洁的形式描述嵌 套命名空间，例如:

```cpp
namespace A::B::C {
	int foo() { return 5; }
}
```

以上代码等同于：

```cpp
namespace A {
	namespace B {
		namespace C {
			int foo() { return 5; }
		}
	}
}
```

很显然前者是一种更简洁的定义嵌套命名空间的方法。除简洁之外，它也更加符合我们已有的语法习惯，比如嵌套类:

```cpp
std::vector<int>::iterator it;
```

在C++20中，我们可以这样定义内联命名空间：

```cpp
namespace A::B::inline C {
	int foo() { return 5; }
}
// 或者
namespace A::inline B::C {
	int foo() { return 5; }
}
```

他们分别等同于：

```cpp
namespace A::B {
	inline namespace C {
		int foo() { return 5; }
	}
}

namespace A {
	inline namespace B {
		namespace C {
			int foo() { return 5; }
		}
	}
}
```

inline 可以出现在除了第一个 namespace 之外的任意 namespace 前。

# auto 占位符

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

# `decltype` 说明符

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
	int operator() () { return 0; }
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

# 函数返回类型后置

## 使用函数返回类型后置声明函数

```cpp
auto foo() -> int {
	return 42;
}
```

在返回类型比较复杂的时候，比如返回一个函数指针类型，返回类型后置可能会是一个不错的选择，例如：

```cpp
int bar_impl(int x) {
	return x;
}

typedef int (*bar) (int);
bar fool() {
	return bar_impl;
}

auto foo2() -> int(*)(int) {
	return bar_impl;
}

int main() {
	auto func = foo2();
	func(58);
}
```


## 推导函数模版返回类型

C++11标准中函数返回类型后置的作用之一是推导函数模板的返回类型，当然前提是需要用到`decltype`说明符。

```cpp
template<class T1, class T2>
auto sum1(T1 t1, T2 t2) -> decltype(t1 + t2) {
	return t1 + t2;
}

int main() {
	auto x1 = sum1(4, 2);
}
```

请注意，decltype(t1 + t2)不能写在函数声明前，编译器在解析返回类型的时候还没解析到参数部分，所以它对t1和t2一无所知，自然会编译失败。

```cpp
decltype(t1 + t2) auto sum1(T1 t1, T2 t2) { ... } // 编译失败，无法识别 t1 和 t2
```

在C++11标准中只用decltype关键字也能写出自动推导返回类型的函数模板，但是函数可读性却差了很多，以下是最容易理解的写法：

```cpp
template<class T1, class T2>
decltype(T1() + T2()) sum2(T1 t1, T2 t2) {
	return t1 + t2;
}

int main() {
	sum2(4, 2);
}
```

这种写法并不通用，它存在一个潜在问题，由于`T1() + T2()`表达式使用了T1和T2类型的默认构造函数，因此编译器要求T1和T2的默认构造函数必须存在，否则会编译失败，比如：

```cpp
class IntWrap {
public:
	IntWrap(int n) : n_(n) {}
	IntWrap operator+ (const IntWrap& other) {
		return IntWrap(n_ + other.n_);
	}
private:
	int n_;
};

int main() {
	sum2(IntWrap(1), IntWrap(2));    // 编译失败，IntWrap没有默认构造函数
}
```

虽然编译器在推导表达式类型的时候并没有真正计算表达式，但是会检查表达式是否正确，所以在推导`IntWrap() + IntWrap()`时会报错。为了解决这个问题，需要既可以在表达式中让T1和T2两个对象求和，又不用使用其构造函数方法，于是就有了以下两个函数模板：

```cpp
template<class T1, class T2>
decltype(*static_cast<T1 *>(nullptr) + *static_cast<T2 *>(nullptr)) sum3(T1 t1, T2 t2) {
	return t1 + t2;
}

template<class T>
T&& declval();

template<class T1, class T2>
decltype(declval<T1>() + declval<T2>()) sum4(T1 t1, T2 t2) {
	return t1 + t2;
}

int main() {
	sum3(IntWrap(1), IntWrap(2));
	sum4(IntWrap(1), IntWrap(2));
}
```

可以看出，虽然这两种方法都能达到函数返回类型后置的效果，但是它们在实现上更加复杂，同时要理解它们也必须有一定的模板元编程的知识。为了让代码更容易被其他人阅读和理解，还是建议使用函数返回类型后置的方法来推导返回类型。

# 右值引用

## 左值和右值

表达式等号左边的值为左值，而表达式右边的值为右值，比如：

```cpp
int x = 1;
int y = 3;
int z = x + y;
```

用表达式等号左右的标准区分左值和右值虽然在一些场景下确实能得到正确结果，但是还是过于简单，有些情况下是无法准确区分左值和右值的，比如：

```cpp
int a = 1;
int b = a;
```

在C++中所谓的左值一般是指一个指向特定内存的具有名称的值（具名对象），它有一个相对稳定的内存地址，并且有一段较长的生命周期。而右值则是不指向稳定内存地址的匿名值（不具名对象），它的生命周期很短，通常是暂时性的。基于这一特征，我们可以用取地址符&来判断左值和右值，能取到内存地址的值为左值，否则为右值。还是以上面的代码为例，因为&a和&b都是符合语法规则的，所以a和b都是左值，而&1在GCC中会给出“lvalue required as unary'&' operand”错误信息以提示程序员&运算符需要的是一个左值。

下面这些情况左值和右值的判断可能是违反直觉的，例如：

```cpp
int x = 1;
int get_val() {
	return x;
}

void set_val(int val) {
	x = val;
}

int main() {
	x++;
	++x;
	int y = get_val();
	set_val(6);
}
```

在上面的代码中，x++和++x虽然都是自增操作，但是却分为不同的左右值。其中x++是右值，因为在后置++操作中编译器首先会生成一份x值的临时复制，然后才对x递增，最后返回临时复制内容。而++x则不同，它是直接对x递增后马上返回其自身，所以++x是一个左值。

```cpp
int *p = &x++; // 编译失败
int *q = &++x; // 编译成功
```

实参6是一个右值，但是进入函数之后形参val却变成了一个左值，我们可以对val使用取地址符，并且不会引起任何问题：

```cpp
void set_val(int val) {
	int *p = &val;
	x = val;
}
```


最后需要强调的是，通常字面量都是一个右值，除字符串字面量以外：

```cpp 
int x = 1;
set_val(6);
auto p = &"hello world";
```

编译器会将字符串字面量存储到程序的数据段中，程序加载的时候也会为其开辟内存空间，所以我们可以使用取地址符&来获取字符串字面量的内存地址。

## 左值引用

左值引用是编程过程中的常用特性之一，它的出现让C++编程在一定程度上脱离了危险的指针。当我们需要将一个对象作为参数传递给子函数的时候，往往会使用左值引用，因为这样可以免去创建临时对象的操作。非常量左值的引用对象很单纯，它们必须是一个左值。对于这一点，常量左值引用的特性显得更加有趣，它除了能引用左值，还能够引用右值，比如：

```cpp
int &x1 = 7; // 编译错误
const int &x = 11; // 编译成功
```

在上面的代码中，第一行代码会编译报错，因为int&无法绑定一个int类型的右值，但是第二行代码却可以编译成功。请注意，虽然在结果上`const int &x = 11`和`const int x = 11`是一样的，但是从语法上来说，前者是被引用了，所以语句结束后11的生命周期被延长，而后者当语句结束后右值11应该被销毁。**虽然常量左值引用可以引用右值的这个特性在赋值表达式中看不出什么实用价值，但是在函数形参列表中却有着巨大的作用**一个典型的例子就是复制构造函数和复制赋值运算符函数，通常情况下我们实现的这两个函数的形参都是一个常量左值引用，例如：

```cpp
class X {
public:
	X() {}
	X(const X&) {}
	X& operator = (const X&) { return *this; }
};

X make_x() {
	return X();
}

int main() {
	X x1;
	X x2(x1);
	X x3(make_x());
	x3 = make_x();
}
```

以上代码可以通过编译，但是如果这里将类 X 的复制构造函数和复制赋值函数形参类型的常量性删除，则`X x3(make_x());`和`x3 =make_x();`这两句代码会编译报错，因为非常量左值引用无法绑定到make_x()产生的右值。**常量左值引用可以绑定右值是一条非常棒的特性，但是它也存在一个很大的缺点——常量性。一旦使用了常量左值引用，就表示我们无法在函数内修改该对象的内容（强制类型转换除外。** 所以需要另外一个特性来帮助我们完成这项工作，它就是右值引用。

## 右值引用

顾名思义，右值引用是一种引用右值且只能引用右值的方法。在语法方面右值引用可以对比左值引用，在左值引用声明中，需要在类型后添加&，而右值引用则是在类型后添加&&，例如：

```cpp
int i = 0;
int &j = i; // 左值引用
int &&k = 11; // 右值引用
```

在上面的代码中，k是一个右值引用，如果试图用k引用变量i，则会引起编译错误。**右值引用的特点之一是可以延长右值的生命周期**，这个对于字面量11可能看不出效果，那么请看下面的例子：

```cpp
#include <iostream>

class X {
public:
	X() { std::cout << "X ctor" << std::endl; }
	X(const X& x) { std::cout << "X copy ctor" << std::endl; }
	~X() { std::cout << "X dtor" << std::endl; }
	void show() { std::cout << "show X" << std::endl; }
};

X make_x() {
	X x1;
	return x1;
}

int main() {
	X &&x2 = make_x();
	x2.show();
}
```

在理解这段代码之前，让我们想一下如果将`X &&x2 = make_x()`这句代码替换为`X x2 = make_x()`会发生几次构造。在没有进行任何优化的情况下应该是3次构造，首先`make_x`函数中`x1`会默认构造一次，然后`return x1`会使用复制构造产生临时对象，接着`X x2 = make_x()`会使用复制构造将临时对象复制到`x2`，最后临时对象被销毁。

以上流程在使用了右值引用以后发生了微妙的变化，让我们编译运行这段代码。请注意，用GCC编译以上代码需要加上命令行参数`-fno-elide-constructors`用于关闭函数返回值优化（RVO）。因为GCC的RVO优化会减少复制构造函数的调用，不利于语言特性实验：

```txt
X ctor
X copy ctor 
X dtor 
show X 
X dtor
```

从运行结果可以看出上面的代码只发生了两次构造。第一次是`make_x`函数中`x1`的默认构造，第二次是`return x1`引发的复制构造。不同的是，由于`x2`是一个右值引用，引用的对象是函数`make_x`返回的临时对象，因此该临时对象的生命周期得到延长，所以我们可以在`X &&x2 = make_x()`语句结束后继续调用`show`函数而不会发生任何问题。对性能敏感的读者应该注意到了，**延长临时对象生命周期并不是这里右值引用的最终目标，其真实目标应该是减少对象复制，提升程序性能。** (少了一次复制构造函数)

## 右值的性能优化空间

通过6.3节的介绍我们知道了很多情况下右值都存储在临时对象中，当右值被使用之后程序会马上销毁对象并释放内存。这个过程可能会引发一个性能问题，例如：

```cpp
#include <iostream>
class BigMemoryPool {
public:
	static const int PoolSize = 4096;
	BigMemoryPool() : pool_(new char[PoolSize]) {} 
	~BigMemoryPool() {
		if (pool_ != nullptr) {
			delete[] poll_;
		}
	}
	
	BigMemoryPool(const BigMemoryPool& other) : pool_(new char[PoolSize]) {
		std::cout << "copy big memory pool." << std::endl;
		memcpy(pool_, other.pool_, PoolSize);
	}
	
private:
	char *pool_;
};

BigMemoryPool get_pool(const BigMemoryPool& pool) {
	return pool;
}

BigMemoryPool make_pool() {
	BigMemoryPool pool;
	return get_pool(pool);
}

int main() {
	BigMemoryPool my_pool = make_pool();
}
```

以上代码同样需要加上编译参数`-fno-elide-constructors`，编译运行程序会在屏幕上输出字符串：

```txt
copy big memory pool.
copy big memory pool.
copy big memory pool.
```

可以看到`BigMemoryPool my_pool = make_pool();`调用了3次复制构造函数。

## 移动语义

仔细分析代码中3次复制构造函数的调用，不难发现**第二次和第三次的复制构造是影响性能的主要原因。在这个过程中都有临时对象参与进来，而临时对象本身只是做数据的复制。** 如果有办法能将临时对象的内存直接转移到my_pool对象中，不就能消除内存复制对性能的消耗吗？好消息是在C++11标准中引入了**移动语义，它可以帮助我们将临时对象的内存移动到my_pool对象中，以避免内存数据的复制。** 让我们简单修改一下`BigMemoryPool`类代码：

```cpp
class BigMemoryPool {
public:
	static const int PoolSize = 4096;
	BigMemoryPool() : pool_(new char[PoolSize]) {}
	~BigMemoryPool() {
		if (pool_ != nullptr) {
			delete[] pool_;
		}
	}
	
	BigMemoryPool(BigMemoryPool&& other) {
		std::cout << "move big memory pool." << std::endl;
		pool_ = other.pool_;
		other.pool_ = nullptr;
	}
	
	BigMemoryPool(const BigMemoryPool& other) : pool_(new char[PoolSize]) {
		std::cout << "copy big memory pool." << std::endl;
		memcpy(pool_, other.pool_, PoolSize);
	}
private:
	char* pool_;
}
```

从构造函数的名称和它们的参数可以很明显地发现其中的区别，对于复制构造函数而言形参是一个左值引用，也就是说**函数的实参必须是一个具名的左值，在复制构造函数中往往进行的是深复制，即在不能破坏实参对象的前提下复制目标对象。** 而移动构造函数恰恰相反，它接受的是一个右值，其核心思想是通过转移实参对象的数据以达成构造目标对象的目的，也就是说实参对象是会被修改的。

编译运行这段代码，其输出结果如下：

```cpp
copy big memory pool;
move big memory pool;
move big memory pool;
```

可以看到后面两次的构造函数变成了移动构造函数，**因为这两次操作中源对象都是右值（临时对象），对于右值编译器会优先选择使用移动构造函数去构造目标对象。** 当移动构造函数不存在的时候才会退而求其次地使用复制构造函数。在移动构造函数中使用了指针转移的方式构造目标对象，所以整个程序的运行效率得到大幅提升。

除移动构造函数能实现移动语义以外，移动赋值运算符函数也能完成移动操作，继续`BigMemoryPool`为例，在这个类中添加移动赋值运算符函数：

```cpp
class BigMemoryPool {
public:
	...
	BigMemoryPool& operator=(BigMemoryPool&& other) {
		std::cout << "move(operator=) big memory pool." << std::endl;
		if (pool_ != nullptr) {
			delete[] pool_;
		}
		pool_ = other.pool_;
		other.pool_ = nullptr;
		return *this;
	}
	
private:

	char *pool_;
};

int main() {
	BigMemoryPool my_pool;
	my_pool = make_pool();
}
```

这段代码编译运行的结果是：

```txt
copy big memory pool.
move big memory pool.
move(operator=) big memory pool
```

可以看到赋值操作`my_pool = make_pool()`调用了移动赋值运算符函数，这里的规则和构造函数一样，即编译器对于赋值源对象是右值的情况会优先调用移动赋值运算符函数，如果该函数不存在，则调用复制赋值运算符函数。

最后有两点需要说明一下。

- 同复制构造函数一样，编译器在一些条件下会生成一份移动构造函数，这些条件包括：**没有任何的复制函数，包括复制构造函数和复制赋值函数；没有任何的移动函数，包括移动构造函数和移动赋值函数；也没有析构函数。** 虽然这些条件严苛得让人有些不太愉快，但是我们也不必对生成的移动构造函数有太多期待，因为编译器生成的移动构造函数和复制构造函数并没有什么区别。

- 虽然使用移动语义在性能上有很大收益，但是却也有一些风险，这些风险来自异常。试想一下，在一个移动构造函数中，如果当一个对象的资源移动到另一个对象时发生了异常，也就是说对象的一部分发生了转移而另一部分没有，这就会造成源对象和目标对象都不完整的情况发生，这种情况的后果是无法预测的。**所以在编写移动语义的函数时建议确保函数不会抛出异常，与此同时，如果无法保证移动构造函数不会抛出异常，可以使用`noexcept`说明符限制该函数。** 这样当函数抛出异常的时候，程序不会再继续执行而是调用`std::terminate`中止执行以免造成其他不良影响。

## 值类别

值类别是C++11标准中新引入的概念，具体来说它是表达式的一种属性，该属性将表达式分为3个类别，它们分别是左值（lvalue）、纯右值（prvalue）和将亡值（xvalue）。

- 所谓泛左值是指一个通过评估能够确定对象、位域或函数的标识的表达式。简单来说，它确定了对象或者函数的标识（具名对象）。

- 而纯右值是指一个通过评估能够用于初始化对象和位域，或者能够计算运算符操作数的值的表达式。

- 将亡值属于泛左值的一种，它表示资源可以被重用的对象和位域，通常这是因为它们接近其生命周期的末尾，另外也可能是经过右值引用的转换产生的。

从本质上说产生将亡值的途径有两种，第一种是使用类型转换将泛左值转换为该类型的右值引用。比如：

```cpp
static_cast<BigMemoryPool&&>(my_pool)
```

第二种在C++17标准中引入，我们称它为临时量实质化，指的是纯右值转换到临时对象的过程。每当纯右值出现在一个需要泛左值的地方时，临时量实质化都会发生，也就是说都会创建一个临时对象并且使用纯右值对其进行初始化，这也符合纯右值的概念，而这里的临时对象就是一个将亡值。

```cpp
struct X {
	int a;
};

int main() {
	int b = X().a;
}
```

在C++17标准之前临时变量是纯右值，只有转换为右值引用的类型才是将亡值。

## 将左值转换为右值

右值引用只能绑定一个右值，如果尝试绑定，左值会导致编译错误：

```cpp
int i = 0;
int &&k = i; // 编译失败
```

在C++11标准中可以在不创建临时值的情况下显式地将左值通过`static_cast`转换为将亡值，通过值类别的内容我们知道将亡值属于右值，所以可以被右值引用绑定。值得注意的是，由于转换的并不是右值，因此它依然有着和转换之前相同的生命周期和内存地址，例如：

```cpp
int i = 0;
int && k = static_cast<int&&>(i); // 编译成功
```

既然这个转换既不改变生命周期也不改变内存地址，那它有什么存在的意义呢？**实际上它的最大作用是让左值使用移动语义**，还是以`BigMemoryPool`为例：

```cpp
BigMemoryPool my_pool1;
BigMemoryPool my_pool2 = my_pool1;
BigMemoryPool my_pool3 = static_cast<BigMemoryPool &&>(my_pool1);
```

由于调用了移动构造函数，`my_pool1`失去了自己的内存数据，后面的代码也不能对`my_pool1`进行操作了。

现在问题又来了，这样单纯地将一个左值数据转换到另外一个左值似乎并没有什么意义。在这个例子中的确如此，这样的转换不仅没有意义，而且如果有程序员在移动构造之后的代码中再次使用`my_pool1`还会引发未定义的行为。正确的使用场景是在一个右值被转换为左值后需要再次转换为右值，最典型的例子是一个右值作为实参传递到函数中。我们在讨论左值和右值的时候曾经提到过，无论一个函数的实参是左值还是右值，其形参都是一个左值，即使这个形参看上去是一个右值引用，例如：

```cpp
void move_pool(BigMemoryPool &&pool) {
	std::cout << "call move_pool" << std::endl;
	BigMemoryPool my_pool(pool);
}

int main() {
	move_pool(make_pool());
}
```

编译运行以上代码输出结果如下：

```txt
copy big memory pool.
move big memory pool.
call move_pool
copy big memory pool.
```

在上面的代码中，`move_pool`函数的实参是`make_pool`函数返回的临时对象，也是一个右值，`move_pool`的形参是一个右值引用，但是在使用形参`pool`构造`my_pool`的时候还是会调用复制构造函数而非移动构造函数。为了让`my_pool`调用移动构造函数进行构造，需要将形参pool强制转换为右值：

```cpp
void move_pool(BigMemoryPool&& pool) {
	std::cout << "call move_pool" << std::endl;
	BigMemoryPool my_pool(static_cast<BigMemoryPool&&>(pool));
}
```

请注意，在这个场景下强制转换为右值就没有任何问题了，因为`move_pool`函数的实参是`make_pool`返回的临时对象，当函数调用结束后临时对象就会被销毁，所以转移其内存数据不会存在任何问题。

在C++11的标准库中还提供了一个函数模板`std::move`帮助我们将左值转换为右值，这个函数内部也是用`static_cast`做类型转换。只不过由于它是使用模板实现的函数，因此会根据传参类型自动推导返回类型，省去了指定转换类型的代码。另一方面从移动语义上来说，使用`std::move`函数的描述更加准确。所以建议读者使用`std::move`将左值转换为右值而非自己使用`static_cast`转换，例如：

```cpp
void move_pool(BigMemoryPool &&pool) {
	std::cout << "call move_pool" << std::endl;
	BigMemoryPool my_pool(std::move(pool));
}
```

## 万能引用和引用折叠

常量左值引用既可以引用左值又可以引用右值，是一个几乎万能的引用，但可惜的是由于其常量性，导致它的使用范围受到一些限制。其实在C++11中确实存在着一个被称为“万能”的引用，它看似是一个右值引用，但其实有着很大区别，请看下面的代码：

```cpp
void foo(int &&i) {} // i 为右值引用

template<class T>
void bar(T &&t) {} // t 为万能引用

int get_val() { return 5; }
int &&x = get_val(); // x 为右值引用
auto &&y = get_val(); // y 为万能引用
```

在上面的代码中，函数`foo`的形参`i`和变量`x`是右值引用，而函数模板的形参`t`和变量`y`则是万能引用。我们知道右值引用只能绑定一个右值，但是万能引用既可以绑定左值也可以绑定右值，甚至`const`和`volatile`的值都可以绑定，例如：

```cpp
int i = 42;
const int j = 11;
bar(i);
bar(j);
bar(get_val());

auto &&x = i;
auto &&y = j;
auto &&z = get_val();
```

万能引用能如此灵活地引用对象，实际上是因为在C++11中添加了一套引用叠加推导的规则——引用折叠。在这套规则中规定了在不同的引用类型互相作用的情况下应该如何推导出最终类型，如下表所示：

| 类模版型 | T实际类型 | 最终类型 |
| ---- | ----- | ---- |
| T&   | R     | R&   |
| T&   | R&    | R&   |
| T&   | R&&   | R&   |
| T&&  | R     | R&&  |
| T&&  | R&    | R&   |
| T&&  | R&&   | R&&  |

只要有左值引用参与进来，最后推导的结果就是一个左值引用。只有实际类型是一个非引用类型或者右值引用类型时，最后推导出来的才是一个右值引用。


值得一提的是，万能引用的形式必须是T&&或者auto&&，也就是说它们必须在初始化的时候被直接推导出来，如果在推导中出现中间过程，则不是一个万能引用，例如：

```cpp
#include <vector>
template<class T>
void foo(std::vector<T> &&t) {}
int main() {
	std::vector<int> v{ 1, 2, 3 }; 
	foo(v);  // 编译错误
}
```

在上面的代码中，foo(v)无法编译通过，因为foo的形参t并不是一个万能引用，而是一个右值引用。因为foo的形参类型是`std::vector<T>&&`而不是`T&&`，所以编译器无法将其看作一个万能引用处理。

## 完美转发

**万能引用最典型的用途被称为完美转发**。在介绍完美转发之前，我们先看一个常规的转发函数模板：

```cpp
#include <iostream>
#include <string>

template<class T>
void show_type(T t) {
	std::cout << typeid(t).name() << std::endl;
}

template<class T>
void normal_forwarding(T t) {
	show_type(t);
}

int main() {
	std::string s = "hello world";
	normal_forwarding(s);
}
```

在上面的代码中，函数`normal_forwarding`是一个常规的转发函数模板，它可以完成字符串的转发任务。但是它的效率却令人堪忧。因为`normal_forwarding`按值转发，也就是说`std::string`在转发过程中会额外发生一次临时对象的复制。其中一个解决办法是将`void normal_forwarding(T t)`替换为`void normal_forwarding(T &t)`，这样就能避免临时对象的复制。不过这样会带来另外一个问题，如果传递过来的是一个右值，则该代码无法通过编译，例如：

```cpp
std::string get_string() {
	return "hi world";
}

normal_forwarding(get_string()); // 编译失败
```

当然，我们还可以将`void normal_forwarding(T &t)`替换为`void normal_forwarding (const T &t)`来解决这个问题，因为常量左值引用是可以引用右值的。但是我们也知道，虽然常量左值引用在这个场景下可以“完美”地转发字符串，但是如果在后续的函数中需要修改该字符串，则会编译错误。所以这些方法都不能称得上是完美转发。

万能引用的出现改变了这个尴尬的局面。上文提到过，对于万能引用的形参来说，**如果实参是给左值，则形参被推导为左值引用；反之如果实参是一个右值，则形参被推导为右值引用**，所以下面的代码无论传递的是左值还是右值都可以被转发，而且不会发生多余的临时复制：

```cpp
#include <iostream>
#include <string>

template<class T>
void show_type(T t) {
	std::cout << typeid(t).name << std::endl;
}

template<class T>
void perfect_forwarding(T &&t) {
	show_type(static_cast<T&&>(t));
}

std::string get_string() {
	return "hi world";
}

int main() {
	std::string s = "hello world";
	perfect_forwarding(s);
	perfect_forwarding(get_string());
}
```

和移动语义的情况一样，显式使用`static_cast`类型转换进行转发不是一个便捷的方法。在C++11的标准库中提供了一个`std::forward`函数模板，在函数内部也是使用`static_cast`进行类型转换，只不过使用`std::forward`转发语义会表达得更加清晰，`std::forward`函数模板的使用方法也很简单：

```cpp
template<class T>
void perfect_forwarding(T &&t) {
	show_type(std::forward<T>(t));
}
```

请注意`std::move`和`std::forward`的区别，其中`std::move`一定会将实参转换为一个右值引用，并且使用`std::move`不需要指定模板实参，模板实参是由函数调用推导出来的。而`std::forward`会根据左值和右值的实际情况进行转发，在使用的时候需要指定模板实参。

## 针对局部变量和右值引用的隐式移动操作

在对旧程序代码升级新编译环境之后，我们可能会发现程序运行的效率提高了，这里的原因一定少不了新标准的编译器在某些情况下将隐式复制修改为隐式移动。

```cpp
#include <iostream>

struct X {
	X() = default;
	X(const X&) = default;
	X(X&&) {
		std::cout << "move ctor";
	}
};

X f(X x) {
	return x;
}

int main() {
	X r = f(X{});
}
```

除此之外，对于局部变量也有相似的规则，只不过大多数时候编译器会采用更加高效的返回值优化代替移动操作，这里我们稍微修改一点f函数：

```cpp
X f() {
	X x;
	return x;
}

int main() {
	X r = f();
}
```

请注意，编译以上代码的时候需要使用`-fno-elide-constructors`选项用于关闭返回值优化。然后运行编译好的程序，会发现`X r = f();`同样调用的是移动构造函数。

在C++20标准中，隐式移动操作针对右值引用和throw的情况进行了扩展，例如：

```cpp
#include <iostream>
#include <string>

struct X {
	X() = default;
	X(const X&) = default;
	X(X&&) {
		std::cout << "move";
	}
};

X(f(X &&x)) {
	return x;
}

int main() {
	X r = f(X{});
}
```

以上代码使用C++20之前的标准编译是不会调用任何移动构造函数的。原因前面也解释过，因为函数`f`的形参`x`是一个左值，对于左值要调用复制构造函数。要实现移动语义，需要将`return x;`修改为`return std::move(x);`。显然这里是有优化空间的，C++20标准规定在这种情况下可以隐式采用移动语义完成赋值。具体规则如下。

可隐式移动的对象必须是一个非易失或一个右值引用的非易失自动存储对象，在以下情况下可以使用移动代替复制。

1. `return`或者`co_return`语句中的返回对象是函数或者`lambda`表达式中的对象或形参。

2. `throw`语句中抛出的对象是函数或`try`代码块中的对象。

实际上`throw`调用移动构造的情况和`return`差不多，我们只需要将上面的代码稍作修改即可：

```cpp
void f() {
	X x;
	throw x;
}

int main() {
	try {
		f();
	}
	catch (...) {
	
	}
}
```

可以看到函数`f`不再有返回值，它通过`throw`抛出`x`，`main`函数用`try-catch`捕获`f`抛出的`x`。这个捕获调用的就是移动构造函数。

# lambda 表达式

## lambda 表达式语法

lambda 表达式的语法定义如下：

```cpp
[ captures ] ( params ) specifiers exception -> ret { body }  
```

例子：

```cpp
#include <iostream>

int main() {
	int x = 5;
	auto foo = [x](int y) -> int { return x * y; }
	std::cout << foo(8) << std::endl;
}
```

- `[captures]` 捕获列表，它可以捕获**当前函数作用域**的零个或多个变量，变量之间用逗号分隔。

- `(params)` 可选参数列表，语法和普通函数的参数列表一样，在不需要参数的时候可以忽略参数列表。

- `specifiers` 可选限定符，C++11中可以用mutable，它允许我们在lambda表达式函数体内改变按值捕获的变量，或者调用非const的成员函数。

- `exception` 可选异常说明符，我们可以使用`noexcept`来指明`lambda`是否会抛出异常。对应的例子中没有使用异常说明符。

- `ret` 可选返回值类型，不同于普通函数，`lambda`表达式使用返回类型后置的语法来表示返回类型，如果没有返回值（`void`类型），可以忽略包括`->`在内的整个部分。另外，我们也可以在有返回值的情况下不指定返回类型，这时编译器会为我们推导出一个返回类型。

- `{body}` ambda表达式的函数体，这个部分和普通函数的函数体一样。

## 捕获列表

### 作用域

捕获列表中的变量存在于两个作用域——**lambda表达式定义的函数作用域以及lambda表达式函数体的作用域。** 前者是为了捕获变量，后者是为了使用变量。另外，标准还规定能捕获的变量必须是一个自动存储类型。简单来说就是非静态的局部变量。让我们看一看下面的例子：

```cpp
int x = 0;

int main() {
	int y = 0;
	static int z = 0;
	auto foo = [x, y, z] {};
}
```

 以上代码可能是无法通过编译的，其原因有两点：第一，变量x和z不是自动存储类型的变量；第二，x不存在于lambda表达式定义的作用域。这里可能无法编译，因为不同编译器对于这段代码的处理会有所不同，比如GCC就不会报错，而是给出警告。

```cpp
#include <iostream>

int x = 1;
int main() {
	int y = 2;
	static int z = 3;
	auto foo = [y] { return x + y + z; };
	std::cout << foo() << std::endl;
}
```

在上面的代码中，虽然我们没有捕获变量x和z，但是依然可以使用它们。进一步来说，如果我们将一个lambda表达式定义在全局作用域，那么lambda表达式的捕获列表必须为空。因为根据上面提到的规则，捕获列表的变量必须是一个自动存储类型，但是全局作用域并没有这样的类型，比如：

```cpp
int x = 1;
auto foo = [] { return x; };
int main() {
	foo();
}
```

### 捕获值和捕获引用

捕获值的语法是在`[]`中直接写入变量名，如果有多个变量，则用逗号分隔，例如：

```cpp
int main() {
	int x = 5, y = 8;
	auto foo = [x, y] { return x * y; }
}
```

捕获值是将函数作用域的x和y的值复制到lambda表达式对象的内部，就如同lambda表达式的成员变量一样。

捕获引用的语法与捕获值只有一个&的区别，要表达捕获引用我们只需要在捕获变量之前加上&，类似于取变量指针。只不过这里捕获的是引用而不是指针，在lambda表达式内可以直接使用变量名访问变量而不需解引用，比如：

```cpp
int main() {
	int x = 5, y = 8;
	auto foo = [&x, &y] { return x * y; };
}
```

```cpp
void bar1() {
	int x = 5, y = 8;
	auto foo = [x, y] {
		x += 1;         // 编译失败，无法改变捕获变量的值
		y += 2;         // 编译失败，无法改变捕获变量的值
		return x * y;	
	};
	std::cout << foo() << std::endl;
}

void bar2() {
	int x = 5, y = 8;
	auto foo = [&x, &y] {
		x += 1;
		y += 2;
		return x * y;
	};
	std::cout << foo() << std::endl;
}
``` 

在上面的代码中函数bar1无法通过编译，原因是我们无法改变捕获变量的值。这就引出了lambda表达式的一个特性：**捕获的变量默认为常量，或者说lambda是一个常量函数（类似于常量成员函数）。**

**使用`mutable`说明符可以移除lambda表达式的常量性，也就是说我们可以在lambda表达式的函数体中修改捕获值的变量了**，例如：

```cpp
void bar3() {
	int x = 5, y = 8;
	auto foo = [x, y] () mutable {
		x += 1;
		y += 2;
		return x * y;
	};
	std::cout << foo() << std::endl;
}
```

**语法规定lambda表达式如果存在说明符，那么形参列表不能省略。**

当lambda表达式捕获值时，表达式内实际获得的是捕获变量的复制，我们可以任意地修改内部捕获变量，但不会影响外部变量。而捕获引用则不同，在lambda表达式内修改捕获引用的变量，对应的外部变量也会被修改：

```cpp
#include <iostream>

int main() {
	int x = 5, y = 8;
	auto foo = [x, &y] () mutable {
		x += 1;
		y += 2;
		std::cout << "lambda x = " << x << ", y = " << y << std::endl;
		return x * y;
	};
	foo();
	std::cout << "call1 x = " << x << ", y = " << y << std::endl;
	foo();
	std::cout << "call2 x = " << x << ", y = " << y << std::endl;
}
```

运行结果如下：

```cpp
lambda x = 6, y = 10
call1 x = 5, y = 10
lambda x = 7, y = 12
call2 x = 5, y = 12
```

捕获值的变量在lambda表达式定义的时候已经固定下来了，无论函数在lambda表达式定义后如何修改外部变量的值，lambda表达式捕获的值都不会变化，例如：

```cpp
#include <iostream>

int main() {
	int x = 5, y = 8l
	auto foo = [x, &y] () mutable {
		x += 1;
		y += 2;
		std::cout << "lambda x = " << x << ", y = " << y << std::endl;
		return x * y;
	};
	x = 9;
	y = 20;
	foo();
}
```

运行结果如下：

```cpp
lambda x = 6, y = 22
```

在上面的代码中，虽然在调用foo之前分别修改了x和y的值，但是捕获值的变量x依然延续着lambda定义时的值，而在捕获引用的变量y被重新赋值以后，lambda表达式捕获的变量y的值也跟着发生了变化。

### 特殊的捕获方法

lambda表达式的捕获列表除了指定捕获变量之外还有3种特殊的捕获方法。

- `[this]` —— 捕获`this`指针，捕获`this`指针可以让我们使用`this`类型的成员变量和函数。

- `[=]` —— 捕获lambda表达式定义作用域的全部变量的值，包括`this`。

- `[&]` —— 捕获lambda表达式定义作用域的全部变量的引用，包括`this`。

捕获 `this` 指针

```cpp
#include <iostream>

class A {
public:
	void print() {
		std::cout << "class A" << std::endl;
	}
	void test() {
		auto foo = [this] {
			print();
			x = 5;
		};
		foo();
	}
private:
	int x;
};

int main() {
	A a;
	a.test();
}
```

在上面的代码中，因为`lambda`表达式捕获了`this`指针，所以可以在`lambda`表达式内调用该类型的成员函数`print`或者使用其成员变量`x`。

捕获全部变量的值或引用：

```cpp
#include <iostream>

int main() {
	int x = 5, y = 8;
	auto foo = [=] { return x * y; };
	std::cout << foo() << std::endl;
}
```

## lambda 表达式的实现原理

让我们从函数对象开始深入探讨`lambda`表达式的实现原理。请看下面的例子：

```cpp
#include <iostream>

class Bar {
public:
	Bar (int x, int y) : x_(x), y_(y) {}
	int operator () (){
		return x_ * y_
	}
private:
	int x_;
	int y_;
};

int main() {
	int x = 5, y = 8;
	auto foo = [x, y] { return x * y; };
	Bar bar(x, y);
	std::cout << "foo() = " << foo() << std::endl;
	std::cout << "bar() = " << bar() << std::endl;
}
```

在上面的代码中，foo是一个lambda表达式，而bar是一个函数对象。它们都能在初始化的时候获取main函数中变量x和y的值，并在调用之后返回相同的结果。这两者比较明显的区别如下。

- 使用lambda表达式不需要我们去显式定义一个类，这一点在快速实现功能上有较大的优势。

- 使用函数对象可以在初始化的时候有更加丰富的操作，例如`Bar bar(x+y, x * y)`，而这个操作在C++11标准的`lambda`表达式中是不允许的。另外，在`Bar`初始化对象的时候使用全局或者静态局部变量也是没有问题的。

**`lambda`表达式的优势在于书写简单方便且易于维护，而函数对象的优势在于使用更加灵活不受限制。**

`lambda`表达式在编译期会由编译器自动生成一个闭包类，在运行时由这个闭包类产生一个对象，我们称它为闭包。在C++中，所谓的闭包可以简单地理解为一个匿名且可以包含定义时作用域上下文的函数对象。

首先，定义一个简单的lambda表达式：

```cpp
#include <iostream>

int main() {
	int x = 5, y = 8;
	auto foo = [=] { return x * y; };
	int z = foo();
}
```

接着，我们用GCC输出其GIMPLE的中间代码：

```txt
main() {
	int D.39253;
	{
		int x;
		int y;
		struct __lambda0 foo;
		typedef struct __lambda0 __lambda0;
		int x;
		try {
			x = 5;
			y = 8;
			foo.__x = x;
			foo.__y = y;
			z = main()::<lambda()>::operator() (&foo);
		}
		finally {
			foo = {CLOBBER};
		}
	}
	D.39253 = 0;
	return D.39253;
}

main::<lambda()>::operator() (const struct __lambda0 * const __closure) {
	int D.39255;
	const int x [value-expr: __closure->__x];
	const int y [value->expr: __closure->__y];
	
	_1 = __closure->__x;
	_2 = __closure->__y;
	D.39255 = _1 * _2;
	return D.39255;
}
```

在某种程度上来说，lambda表达式是C++11给我们提供的一块语法糖而已，lambda表达式的功能完全能够手动实现，而且如果实现合理，代码在运行效率上也不会有差距，只不过实用lambda表达式让代码编写更加轻松。

## 无状态 lambda 表达式

C++标准对于无状态的lambda表达式有着特殊的照顾，即它可以隐式转换为函数指针，例如：

```cpp
void f(void(*)()) {}
void g() { f([] {}); } // 编译成功
```

在上面的代码中，lambda表达式`[] {}`隐式转换为`void(*)()`类型的函数指针。同样，看下面的代码：

```cpp
void f(void(&)()) {}
void g() { f(*[] ()); }
```

这段代码也可以顺利地通过编译。我们经常会在STL的代码中遇到 `lambda` 表达式的这种应用。

## 在 STL 中使用 lambda 表达式


在有了lambda表达式以后，我们可以直接在STL算法函数的参数列表内实现辅助函数，例如：

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
	std::vector<int> x = {1, 2, 3, 4, 5};
	std::cout << *std::find_if(x.cbegin(), x.cend(), [] (int i) { return (i % 3) == 0}) << std::endl;
```

## 广义捕获

所谓广义捕获实际上是两种捕获方式，第一种称为简单捕获，这种捕获就是我们在前文中提到的捕获方法，即`[identifier]`、`[&identifier]`以及`[this]`等。第二种叫作初始化捕获，这种捕获方式是在C++14标准中引入的，它解决了简单捕获的一个重要问题，**即只能捕获lambda表达式定义上下文的变量，而无法捕获表达式结果以及自定义捕获变量名**，比如：

```cpp
int main() {
	int x = 5;
	auto foo = [x = x + 1] { return x; };
}
```

以上在C++14标准之前是无法编译通过的，因为C++11标准只支持简单捕获。而C++14标准对这样的捕获进行了支持，在这段代码里捕获列表是一个赋值表达式，不过这个赋值表达式有点特殊，因为它通过等号跨越了两个作用域。等号左边的变量x存在于lambda表达式的作用域，而等号右边x存在于main函数的作用域。

```cpp
int main() {
	int x = 5;
	auto foo = [r = x + 1] { return r; };
}
```

初始化捕获在某些场景下是非常实用的，这里举两个例子，**第一个场景是使用移动操作减少代码运行的开销**，例如：

```cpp
#include <string>

int main() {
	std::string x = "hello c++ ";
	auto foo = [x = std::move(x)] { return x + "world"; };
}
```

上面这段代码使用std::move对捕获列表变量x进行初始化，这样避免了简单捕获的复制对象操作，代码运行效率得到了提升。

第二个场景是在异步调用时复制this对象，防止lambda表达式被调用时因原始this对象被析构造成未定义的行为，比如：

```cpp
#include <iostream>
#include <future>

class Work {
private:
	int value;
public:
	Work() : value(42) {}
	std::future<int> spawn() {
		return std::async([=]() -> int { return value; });
	}
};

std::future<int> foo() {
	Work tmp;
	return tmp.spawn();
}

int main() {
	std::future<int> f = foo();
	f.wait();
	std::cout << "f.get() = " << f.get() << std::endl;
}
```

输出结果如下：

```txt
f.get() = 32766
```

这里我们期待f.get()返回的结果是42，而实际上返回了32766，这就是一个未定义的行为，它造成了程序的计算错误，甚至有可能让程序崩溃。为了解决这个问题，我们引入初始化捕获的特性，将对象复制到lambda表达式内，让我们简单修改一下spawn函数：

```cpp
class Work {
private:
	int value;
public:
	Work() : value(42) {}
	std::future<int> spawn() {
		return std::async([=, tmp=*this]() -> int { return tmp.value; });
	}
};
```

以上代码使用初始化捕获，将`*this`复制到`tmp`对象中，然后在函数体内返回`tmp`对象的`value`。由于整个对象通过复制的方式传递到`lambda`表达式内，因此即使`this`所指的对象析构了也不会影响`lambda`表达式的计算。编译运行修改后的代码，程序正确地输出`f.get() =42`。

## 泛型 lambda 表达式

C++14标准让`lambda`表达式具备了模版函数的能力，我们称它为泛型`lambda`表达式。虽然具备模版函数的能力，但是它的定义方式却用不到`template`关键字。实际上泛型`lambda`表达式语法要简单很多，我们只需要使用`auto`占位符即可，例如：

```cpp
int main() {
	auto foo = [](auto a) { return a; };
	int three = foo(3);
	char const* hello = foo("hello");
}
```

## 常量 `lambda` 表达式和捕获 `*this`

C++17标准对`lambda`表达式同样有两处增强，一处是常量`lambda`表达式，另一处是对捕获`*this`的增强。其中常量`lambda`表达式的主要特性体现在`constexpr`关键字上，请阅读`constexpr`的有关章节来掌握常量`lambda`表达式的特性，这里主要说明一下对于捕获`this`的增强。

```cpp
class Work {
private:
	int value;
public:
	Work() : value(42) {}
	std::future<int> spawn() {
		return std::async([=, *this]() -> int { return value; });
	}
};
```

在上面的代码中没有再使用`tmp=*this`来初始化捕获列表，而是直接使用`*this`。在`lambda`表达式内也没有再使用`tmp.value`而是直接返回了`value`。编译运行这段代码可以得到预期的结果42。从结果可以看出，`[*this]`的语法让程序生成了一个`*this`对象的副本并存储在 `lambda` 表达式内，可以在`lambda`表达式内直接访问这个复制对象的成员，消除了之前`lambda`表达式需要通过`tmp`访问对象成员的尴尬。

## 捕获 `[=, this]`

在C++20标准中，又对`lambda`表达式进行了小幅修改。这一次修改没有加强`lambda`表达式的能力，而是让`this`指针的相关语义更加明确。我们知道`[=]`可以捕获`this`指针，相似的，`[=,*this]`会捕获`this`对象的副本。但是在代码中大量出现`[=]`和`[=,*this]`的时候我们可能很容易忘记前者与后者的区别。为了解决这个问题，在C++20标准中引入了`[=, this]`捕获this指针的语法，它实际上表达的意思和`[=]`相同，目的是让程序员们区分它与`[=,*this]`的不同：

```cpp
[=, this]{}; // c++17 编译报错或者报警告，c++20成功编译
```

虽然在C++17标准中认为`[=, this]{};`是有语法问题的，但是实践中GCC和CLang都只是给出了警告而并未报错。另外，在C++20标准中还特别强调了要用`[=, this]`代替`[=]`，如果用GCC编译下面这段代码：

```cpp
template <class T>
void g(T) {}

struct Foo {
	int n = 0;
	void f(int a) {
		g([=](int k) { return n + a * k; });
	}
};
```

编译器会输出警告信息，表示标准已经不再支持使用`[=]`隐式捕获`this`指针了，提示用户显式添加`this`或者`*this`。最后值得注意的是，同时用两种语法捕获`this`指针是不允许的，比如：

```cpp
[this, *this]{};
```

## 模版语法的泛型 lambda 表达式

我们讨论了C++14标准中`lambda`表达式通过支持`auto`来实现泛型。大部分情况下，这是一种不错的特性，但不幸的是，这种语法也会使我们难以与类型进行互动，对类型的操作变得异常复杂。用提案文档的举例来说：

```cpp
template <typename T> struct is_std_vector : std::false_type { };
template <typename T> struct is_std_vector<std::vector<T>> : std::true_type { };
auto f = [](auto vector) {
	static_assert(is_std_vector<decltype(vector)>::value, "");
};
```

普通的函数模板可以轻松地通过形参模式匹配一个实参为vector的容器对象，但是对于lambda表达式，auto不具备这种表达能力，所以不得不实现is_std_vector，并且通过static_assert来辅助判断实参的真实类型是否为vector。在C++委员会的专家看来，把一个本可以通过模板推导完成的任务交给static_assert来完成是不合适的。除此之外，这样的语法让获取vector存储对象的类型也变得十分复杂，比如：

```cpp
auto f = [](auto vector) {
	using T = typename decltype(vector)::value_type;
	// ...
};
```

当然，能这样实现已经是很侥幸了。我们知道vector容器类型会使用内嵌类型value_type表示存储对象的类型。但我们并不能保证面对的所有容器都会实现这一规则，所以依赖内嵌类型是不可靠的。

进一步来说，`decltype(obj)`有时候并不能直接获取我们想要的类型。

```cpp
auto f = [](const auto& x) {
	using T = decltype(x);
	T copy = x; // 可以编译，但是语义错误
	using Iterator = typename T::iterator: // 编译错误
};
std::vector<int> v;
f(v);
```

请注意，在上面的代码中，`decltype(x)`推导出来的类型并不是`std::vector` ，而是`const std::vector &`，所以`T copy = x;`不是一个复制而是引用。对于一个引用类型来说，`T::iterator`也是不符合语法的，所以编译出错。在提案文档中，作者很友好地给出了一个解决方案，他使用了STL的`decay`，这样就可以将类型的cv以及引用属性删除，于是就有了以下代码：

```cpp
auto f = [](const auto& x) {
	using T = std::decay_t<decltype(x)>;
	T copy = x;
	using Iterator = typename T::iterator;
};
```

问题虽然解决了，但是要时刻注意auto，以免给代码带来意想不到的问题，况且这都是建立在容器本身设计得比较完善的情况下才能继续下去的。

鉴于以上种种问题，C++委员会决定在C++20中添加模板对lambda的支持，语法非常简单：

```cpp
[]<typename T>(T t) {}
```

于是，上面那些让我们为难的例子就可以改写为：

```cpp
auto f = []<typename T>(std::vector<T> vector) {
	// ...
};

auto f = []<typename T>（T const& x) {
	T copy = x;
	using Iterator = typename T::iterator;
};
```

## 可构造和可赋值的无状态 lambda 表达式

无状态lambda表达式可以转换为函数指针，但遗憾的是，在C++20标准之前无状态的lambda表达式类型既不能构造也无法赋值，这阻碍了许多应用的实现。举例来说，我们已经了解了像std::sort和std::find_if这样的函数需要一个函数对象或函数指针来辅助排序和查找，这种情况我们可以使用lambda表达式完成任务。但是如果遇到std::map这种容器类型就不好办了，因为std::map的比较函数对象是通过模板参数确定的，这个时候我们需要的是一个类型：

```cpp
auto greater = [](auto x, auto y) { return x > y; };
std::map<std::string, int, decltype(greater)> mymap;
```

这段代码的意图很明显，它首先定义了一个无状态的`lambda`表达式`greater`，然后使用`decltype(greater)`获取其类型作为模板实参传入模板。这个想法非常好，但是在C++17标准中是不可行的，因为`lambda`表达式类型无法构造。编译器会明确告知，`lambda`表达式的默认构造函数已经被删除了。

除了无法构造，无状态的lambda表达式也没办法赋值，比如：

```cpp
auto greater = [](auto x, auto y) { return x > y; };
std::map<std::string, int, decltype(greater)> mymap1, mymap2;
mymap1 = mymap2;
```

这里mymap1 = mymap2;也会被编译器报错，原因是复制赋值函数也被删除了。为了解决以上问题，C++20标准允许了无状态lambda表达式类型的构造和赋值，所以使用C++20标准的编译环境来编译上面的代码是可行的。

# 非静态数据成员默认初始化

## 使用默认初始化

在C++11以前，对非静态数据成员初始化需要用到初始化列表，当类的数据成员和构造函数较多时，编写构造函数会是一个令人头痛的问题：

```cpp
class X {
public:
	X() : a_(0), b_(0.), c_("hello world") {}
	X(int a) : a_(a), b_(0.), c_("hello world") {}
	X(double b) : a_(0), b_(b), c_("hello world") {}
	X(const std::string &c) : a_(0), b_(0.), c_(c) {}

private:
	int a_;
	double b_;
	std::string c_;
};
```

C++11标准提出了新的初始化方法，即在声明非静态数据成员的同时直接对其使用`=`或者`{}`初始化。在此之前只有类型为整型或者枚举类型的常量静态数据成员才有这种声明默认初始化的待遇：

```cpp
class X {
public:
	X() {}
	X(int a) : a_(a) {}
	X(double b) : b_(b) {}
	X(const std::string &c) : c_(c) {}
private:
	int a_ = 0;
	double b_{ 0. };
	std::string c_{ "hello world" };
};
```

在初始化的优先级上有这样的规则，初始化列表对数据成员的初始化总是优先于声明时默认初始化。

最后来看一看非静态数据成员在声明时默认初始化需要注意的两个问题。

- 不要使用括号()对非静态数据成员进行初始化，因为这样会造成解析问题，所以会编译错误。

- 不要用auto来声明和初始化非静态数据成员，虽然这一点看起来合理，但是C++并不允许这么做。

```cpp
struct X {
	int a(5);   // 编译错误，不能使用()进行默认初始化
	auto b = 8; // 编译错误，不能使用 auto 声明和初始化非静态数据成员	
};
```

## 位域的默认初始化

在C++11标准提出非静态数据成员默认初始化方法之后，C++20标准又对该特性做了进一步扩充。在C++20中我们可以对数据成员的位域进行默认初始化了，例如：

```cpp
struct S {
	int y : 8 = 11;
	int z : 4 {7};
};
```

在上面的代码中，int数据的低8位被初始化为11，紧跟它的高4位被初始化为7。

位域的默认初始化语法很简单，但是也有一个需要注意的地方。当表示位域的常量表达式是一个条件表达式时我们就需要警惕了，例如：

```cpp
int a;
struct S2 {
	int y : true ? 8 : a = 42;
	int z : 1 || new int { 0 };
};
```

请注意，这段代码中并不存在默认初始化，因为最大化识别标识符的解析规则让`=42`和`{0}`不可能存在于解析的顶层。于是以上代码会被认为是：

```cpp
int a;
struct S2 {
	int y : (true ? 8 : a = 42);
	int z : (1 || new int { 0 });
};
```

所以我们可以通过使用括号明确代码被解析的优先级来解决这个问题：

```cpp
int a;
struct S2 {
	int y : (true ? 8 : a) = 42;
	int z : (1 || new int) { 0 };
};
```

通过以上方法就可以对`S2::y`和`S2::z`进行默认初始化了。

# 列表初始化

## 回顾变量初始化

在介绍列表初始化之前，让我们先回顾一下初始化变量的传统方法。其中常见的是使用括号和等号在变量声明时对其初始化，例如：

```cpp
struct C {
	C(int a) {}
};

int main() {
	int x = 5;
	int x1(8);
	C x2 = 4;
	C x3(4);
}
```

一般来说，我们称使用括号初始化的方式叫作直接初始化，而使用等号初始化的方式叫作拷贝初始化（复制初始化）。请注意，这里使用等号对变量初始化并不是调用等号运算符的赋值操作。实际情况是，等号是拷贝初始化，调用的依然是直接初始化对应的构造函数，只不过这里是隐式调用而已。如果我们将`C(int a)`声明为`explicit`，那么`C x2 = 4`就会编译失败。

**使用括号和等号只是直接初始化和拷贝初始化的代表**，还有一些经常用到的初始化方式也属于它们。**比如new运算符和类构造函数的初始化列表就属于直接初始化，而函数传参和return返回则是拷贝初始化。** 前者比较好理解，后者可以通过具体的例子来理解：

```cpp
#include <map>
struct C {
	C(int a) {}
};

void foo(C c) {} 
C bar() {
	return 5;
}

int main() {
	foo(8);     // 拷贝初始化
	C c = bar(); // 拷贝初始化
}
```

## 使用列表初始化

C++11标准引入了列表初始化，它使用大括号{}对变量进行初始化，和传统变量初始化的规则一样，它也区分为直接初始化和拷贝初始化，例如：

```cpp
#include <string>

struct C {
	C(std::string a, int b) {}
	C(int a) {}
};

void foo(C) {}
C bar() {
	return { "world", 5 };
}

int main() {
	int x = {5};        // 拷贝初始化
	int x1{8};          // 直接初始化
	C x2 = {4};         // 拷贝初始化
	C x3{2};            // 直接初始化
	foo({8});           // 拷贝初始化
	foo({"hello", 8});  // 拷贝初始化
	C x4 = bar();       // 拷贝初始化
	C *x5 = new C{ "hi", 42}  // 直接初始化
}
```

**有时候我们并不希望编译器进行隐式构造，这时候只需要在特定构造函数上声明explicit即可。**

讨论使用大括号初始化变量就不得不提用大括号初始化数组，例如`int x[] = { 1,2,3,4,5 }`。不过遗憾的是，这个特性无法使用到STL的`vector`、`list`等容器中。想要初始化容器，我们不得不编写一个循环来完成初始化工作。现在，列表初始化将程序员从这个问题中解放了出来，我们可以使用列表初始化对标准容器进行初始化了，例如：

```cpp
#include <vector>
#include <list>
#include <set>
#include <map>
#include <string>

int main() {
	int x[] = { 1, 2, 3, 4, 5 };
	int x1[]{ 1, 2, 3, 4, 5 };
	std::vector<int> x2{ 1, 2, 3, 4, 5 };
	std::vector<int> x3 = { 1, 2, 3, 4, 5 };
	std::list<int> x4{ 1, 2, 3, 4, 5 };
	std::list<int> x5 = { 1, 2, 3, 4, 5 };
	std::set<int> x6{ 1, 2, 3, 4, 5 };
	std::set<int> x7 = { 1, 2, 3, 4, 5 };
	std::map<std::string, int> x8{ {"bear", 4}, {"cassowary", 2}, {"tiger", 7}};
	std::map<std::string, int> x9 = { {"bear", 4}, {"cassowary", 2}, {"tiger", 7}};
}
```

## `std::initializer_list`详解

标准容器之所以能够支持列表初始化，离不开编译器支持的同时，它们自己也必须满足一个条件：**支持`std::initializer_list`为形参的构造函数**。`std::initializer_list`简单地说就是一个支持`begin`、`end`以及`size`成员函数的类模板，有兴趣的读者可以翻阅STL的源代码，然后会发现无论是它的结构还是函数都直截了当。编译器负责将列表里的元素（大括号包含的内容）构造为一个`std::initializer_list` 的对象，然后寻找标准容器中支持`std:: initializer_list`为形参的构造函数并调用它。而标准容器的构造函数的处理就更加简单了，它们只需要调用`std::initializer_list`对象的`begin`和`end`函数，在循环中对本对象进行初始化。


通过了解原理能够发现，支持列表初始化并不是标准容器的专利，我们也能写出一个支持列表初始化的类，需要做的只是添加一个以`std::initializer_list`为形参的构造函数罢了，比如下面的例子：

```cpp
#include <iostream>
#include <string>

struct C {
	C(std::initializer_list<std::string> a) {
		for (const std::string* item = a.begin(); item != a.end(); item++) {
			std::cout << *item << " ";
		}
		std::cout << std::endl;
	}
};

int main() {
	C c{ "hello", "c++", "world" };
}
```

上面这段代码实现了一个支持列表初始化的类 `C`，类 `C` 的构造函数为`C(std:: initializer_list<std::string> a)`，这是支持列表初始化所必需的，值得注意的是，`std::initializer_list`的`begin`和`end`函数并不是返回的迭代器对象，而是一个常量对象指针`const T *`。本着刨根问底的精神，让我们进一步探究编译器对列表的初始化处理：

```cpp
#include <iostream>
#include <string>

struct C {
	C(std::initializer_list<std::string> a) {
		for (const std::string* item = a.begin(); item != a.end(); item++) {
			std::cout << item << " ";
		}
		std::cout << std::endl;
	}
};

int main() {
	C c{ "hello", "c++", "world" };
	std::cout << "sizeof(std::string) = " << std::hex << sizeof(std::string) << std::endl;
}
```

运行输出结果如下：

```txt
0x77fdd0 0x77fdf0 0x77fe10
sizeof(std::string) = 20
```

以上代码输出了`std::string`对象的内存地址以及单个对象的大小（不同编译环境的`std::string`实现方式会有所区别，其对象大小也会不同，这里的例子是使用GCC编译的，`std::string`对象的大小为0x20）。仔细观察3个内存地址会发现，它们的差别正好是`std::string`所占的内存大小。于是我们能推断出，编译器所进行的工作大概是这样的：

```cpp
const std::string __a[3] = {std::string{"hello"}, std::string{"c++"}, std::string{"world"}};
C c(std::initializer_list<std::string>(__a, __a + 3));
```

另外，有兴趣的读者不妨用GCC对上面这段代码生成中间代码GIMPLE，不出意外会发现类似这样的中间代码：

```cpp
main() {
	struct initializer_list D.40094;
	const struct basic_string D.36430[3];
	...
	std::__cxx11::basic_string<char>::basic_string (&D.36430[0], "hello", &D.36424);
	...
	std::__cxx11::basic_string<char>::basic_string (&D.36430[1], "c++", &D.36426);
	...
	std::__cxx11::basic_string<char>::basic_string (&D.36430[2], "world", &D.36428);
	...
	D.40094._M_array = &D.36430;
	D.40094._M_len = 3;
	C::C (&c, D.40094);
	...
}
```

## 使用列表初始化的注意事项

### 隐式缩窄转换问题

隐式缩窄转换是在编写代码中稍不留意就会出现的，而且它的出现并不一定会引发错误，甚至有可能连警告都没有，所以有时候容易被人们忽略，比如：

```cpp
int x = 12345;
char y = x;
```

这段代码中变量y的初始化明显是一个隐式缩窄转换，这在传统变量初始化中是没有问题的，代码能顺利通过编译。但是如果采用列表初始化，比如`char z{ x }`，根据标准编译器通常会给出一个错误，MSVC和CLang就是这么做的，而GCC有些不同，它只是给出了警告。

现在问题来了，在C++中哪些属于隐式缩窄转换呢？在C++标准里列出了这么4条规则。

- 从浮点类型转换整数类型。

- 从`long double`转换到`double`或`float`，或从`double`转换到`float`，除非转换源是常量表达式以及转换后的实际值在目标可以表示的值范围内。

- 从整数类型或非强枚举类型转换到浮点类型，除非转换源是常量表达式，转换后的实际值适合目标类型并且能够将生成目标类型的目标值转换回原始类型的原始值。

- 从整数类型或非强枚举类型转换到不能代表所有原始类型值的整数类型，除非源是一个常量表达式，其值在转换之后能够适合目标类型。

```cpp
int x = 999;
const int y = 999;
const int z = 99;
const double cdb = 99.9;
double db = 99.9;
char c1 = x;  // 编译成功，传统变量初始化支持隐式缩窄转换
char c2{ x }; // 编译失败，可能是隐式缩窄转换，对应规则4
char c3{ y }; // 编译失败，确定是隐式缩窄转换，999超出char能够适应的范围，对应规则4
char c4{ z }; // 编译成功，99在char能够适应的范围内，对应规则4
unsigned char uc1 = { 5 }; // 编译成功，5在unsigned char能够适应的范围内，对应规则4
unsigned char uc2 = { -1 }; // 编译失败，unsigned char不能够适应负数，对应规则4
unsigned int uil = { -1 }; // 编译失败，signed int不能够适应-1所对应的unsigned int，通常是4294967295，对应规则4
int ii = { 2.0 }; // 编译失败，int不能适应浮点范围，对应规则1
float f1{ x }; // 编译失败，float可能无法适应整数或者互相转换，对应规则3
float f2{ 7 }; // 编译成功，7能够适应float，且float也能转换回整数7，对应规则3
float f3{ cdb }; // 编译成功，99.9能适应float，对应规则2
float f4{ db }; // 编译失败，可能是隐式缩窄转无法表达double，对应规则2
```

### 列表初始化的优先级问题

列表初始化既可以支持普通的构造函数，也能够支持以`std::initializer_list`为形参的构造函数。如果这两种构造函数同时出现在同一个类里，那么编译器会如何选择构造函数呢？比如：

```cpp
std::vector<int> x1(5, 5);
std::vector<int> x2{ 5, 5 };
```

如果有一个类同时拥有满足列表初始化的构造函数，且其中一个是以`std::initializer_list`为参数，那么编译器将优先以`std::initializer_ list`为参数构造函数。由于这个特性的存在，我们在编写或阅读代码的时候就一定需要注意初始化代码的意图是什么，应该选择哪种方法对变量初始化。

最后让我们回头看一看9.2节中没有解答的一个问题，`std::map<std:: string, int> x8{ {"bear",4},{"cassowary",2}, {"tiger",7} }`中两个层级的列表初始化分别使用了什么构造函数。其实答案已经非常明显了，内层`{"bear",4}`、`{"cassowary",2}`和`{"tiger",7}`都隐式调用了`std::pair`的构造函数`pair(const T1& x, const T2& y)`，而外层的`{…}`隐式调用的则是`std::map`的构造函数`map(std::initializer_list<value_ type>init, constAllocator&)`。

## 指定初始化

为了提高数据成员初始化的可读性和灵活性，C++20标准中引入了指定初始化的特性。该特性允许指定初始化数据成员的名称，从而使代码意图更加明确。让我们看一看示例：

```cpp
struct Point {
	int x;
	int y;
};

Point p { .x = 4, .y = 2 };
```

虽然在这段代码中Point的初始化并不如Point p{ 4, 2 };方便，但是这个例子却很好地展现了指定初始化语法。实际上，当初始化的结构体的数据成员比较多且真正需要赋值的只有少数成员的时候，这样的指定初始化就非常好用了：

```cpp
struct Point3D {
	int x;
	int y;
	int z;
};

Point3D p{ .z = 3; }; // x = 0, y = 0
```

最后需要注意的是，并不是什么对象都能够指定初始化的。

**它要求对象必须是一个聚合类型，例如下面的结构体就无法使用指定初始化：**

```cpp
struct Point3D {
	Point3D() {}
	int x;
	int y;
	int z;
};

Point3D p{ .z = 3 }; // 编译失败，Point3D不是一个聚合类型
```

如果不能提供构造函数，那么我们希望数据成员x和y的默认值不为0的时候应该怎么做？不要忘了，从C++11开始我们有了非静态成员变量直接初始化的方法，比如当希望Point3D的默认坐标值都是100时，代码可以修改为：

```cpp
struct Point3D {
	int x = 100;
	int y = 100;
	int z = 100;
};

Point3D p{ .z = 3 }; // x = 100, y = 100, z = 3;
```

**指定的数据成员必须是非静态数据成员。这一点很好理解，静态数据成员不属于某个对象。**

**每个非静态数据成员最多只能初始化一次：**

```cpp
Point p{ .y = 4, .y = 2 }; // 编译失败，y不能初始化多次
```

**非静态数据成员的初始化必须按照声明的顺序进行。** 请注意，这一点和C语言中指定初始化的要求不同，在C语言中，乱序的指定初始化是合法的，但C++不行。其实这一点也很好理解，因为C++中的数据成员会按照声明的顺序构造，按照顺序指定初始化会让代码更容易阅读：

```cpp
Point p{ .y = 4, .x = 2 }; // C++ 编译失败，C编译没问题
```

**针对联合体中的数据成员只能初始化一次，不能同时指定：**

```cpp
union u {
	int a;
	const char* b;
};

u f = { .a = 1};     // 编译成功
u g = { .b = "asdf" };   // 编译成功
u h = { .a = 1, .b = "asdf" };  // 编译失败，同时指定初始化联合体中的多个数据成员
```

**不能嵌套指定初始化数据成员。虽然这一点在C语言中也是允许的，但是C++标准认为这个特性很少有用，所以直接禁止了：**

```cpp
struct Line {
	Point a;
	Point b;
};

Line l{ .a.y = 5; }; // 编译失败, .a.y = 5访问了嵌套成员，不符合C++标准
```

当然，如果确实想嵌套指定初始化，我们可以换一种形式来达到目的：

```cpp
Line l{ .a {.y = 5} };
```

**在C++20中，一旦使用指定初始化，就不能混用其他方法对数据成员初始化了，而这一点在C语言中是允许的：**

```cpp
Point p{ .x = 2, 3 }; // 编译失败，混用数据成员的初始化
```

最后再来了解一下指定初始化在C语言中处理数组的能力，当然在C++中这同样是被禁止的：

```cpp
int arr[3] = { [1] = 5 }; // 编译失败
```

C++标准中给出的禁止理由非常简单，它的语法和lambda表达式冲突了。

# 默认和删除函数

## 类的特殊成员函数

在定义一个类的时候，我们可能会省略类的构造函数，因为C++标准规定，在没有自定义构造函数的情况下，编译器会为类添加默认的构造函数。像这样有特殊待遇的成员函数一共有6个（C++11以前是4个），具体如下。

- 默认构造函数。

- 析构函数。

- 复制构造函数。

- 复制赋值运算符函数。

- 移动构造函数（C++11新增）。

- 移动赋值运算符函数（C++11新增）。

添加默认特殊成员函数的这条特性非常实用，它让程序员可以有更多精力关注类本身的功能而不必为了某些语法特性而分心，同时也避免了让程序员编写重复的代码，比如：

```cpp
#include <string>
#include <vector>
class City {
	std::string name;
	std::vector<std::string> street_name;
};

int main() {
	City a, b;
	a = b;
}
```

在上面的代码中，我们虽然没有为City类添加复制赋值运算符函数`City:: operator= (const City &)`，但是编译器仍然可以成功编译代码，并且在运行过程中正确地调用`std::string`和`std::vector<std::string>`的复制赋值运算符函数。假如编译器没有提供这条特性，我们就不得不在编写类的时候添加以下代码：

```cpp
City& City::operator = (const City & other) {
	name = other.name;
	street_name = other.street_name;
	return *this;
}
```

很明显，编写这段代码除了满足语法的需求以外没有其他意义，很庆幸可以把这件事情交给编译器去处理。不过还不能高兴得太早，因为该特性的存在也给我们带来了一些麻烦。

- **声明任何构造函数都会抑制默认构造函数的添加。**

- **一旦用自定义构造函数代替默认构造函数，类就将转变为非平凡类型。**

- **没有明确的办法彻底禁止特殊成员函数的生成（C++11之前）。**

```cpp
#include <string>
#include <vector>
class City {
	std::string name;
	std::vector<std::string> street_name;
public:
	City(const char* n) : name(n) {}
};

int main() {
	City a("wuhan");
	City b; // 编译失败，自定义构造函数抑制了默认构造函数
	b = a;
}
```

以上代码由于添加了构造函数`City(const char *n)`，导致编译器不再为类提供默认构造函数，因此在声明对象b的时候出现编译错误，为了解决这个问题我们不得不添加一个无参数的构造函数：

```cpp
class City {
	std::string name;
	std::vector<std::string> street_name;
public:
	City(const char* n) : name(n) { }
	City() {} // 新添加的构造函数
};
```

可以看到这段代码新添加的构造函数什么也没做，但却必须定义。乍看虽然做了一些多此一举的工作，但是毕竟也能让程序重新编译和运行，问题得到了解决。真的是这样吗？事实上，我们又不知不觉地陷入另一个麻烦中，请看下面的代码：

```cpp
class Trivial {
	int i;
public:
	Trivial(int n) : i(n), j(n) {}
	Trivial() {}
	int j;
};

int main() {
	Trivial a(5);
	Trivial b;
	b = a;
	std::cout << "std::is_trivial_v<Trivial> : " << std::is_trivial_v<Trivial> << std::endl;
}
```

上面的代码中有两个动作会将`Trivial`类的类型从一个平凡类型转变为非平凡类型。第一是定义了一个构造函数`Trivial(int n)`，它导致编译器抑制添加默认构造函数，于是`Trivial`类转变为非平凡类型。第二是定义了一个无参数的构造函数，同样可以让`Trivial`类转变为非平凡类型。

最后一个问题大家肯定也都遇到过，举例来说，**有时候我们需要编写一个禁止复制操作的类，但是过去C++标准并没有提供这样的能力。** 聪明的程序员通过将复制构造函数和复制赋值运算符函数声明为private并且不提供函数实现的方式，间接地达成目的。为了使用方便，boost库也提供了noncopyable类辅助我们完成禁止复制的需求。

不过就如前面的问题一样，虽然能间接地完成禁止复制的需求，但是这样的实现方法并不完美。比如，**友元就能够在编译阶段破坏类对复制的禁止。** 这里可能会有读者反驳，虽然友元能够访问私有的复制构造函数，但是别忘了，我们并没有实现这个函数，也就是说程序最后仍然无法运行。没错，**程序最后会在链接阶段报错**，原因是找不到复制构造函数的实现。但是这个报错显然来得有些晚，试想一下，如果面临的是一个巨大的项目，有不计其数的源文件需要编译，那么编译过程将非常耗时。如果某个错误需要等到编译结束以后的链接阶段才能确定，那么修改错误的时间代价将会非常高，所以**我们还是更希望能在编译阶段就找到错误。**

还有一个典型的例子，禁止重载函数的某些版本，考虑下面的例子：

```cpp
class Base {
	void foo(long &);
public:
	void foo(int) {}
};

int main() {
	Base b;
	long l = 5;
	b.foo(8);
	b.foo(l); // 编译错误
}
```

假设现在我们需要继承Base类，并且实现子类的foo函数；另外，还想沿用基类Base的foo函数，于是这里使用using说明符将Base的foo成员函数引入子类，代码如下：

```cpp
class Base {
	void foo(long &);
public:
	void foo(int) {}
};

class Derived : public Base {
public:
	using Base::foo;
	void foo(const char*) {}
};

int main() {
	Der   ived d;
	d.foo("hello");
	d.foo(5);
}
``` 

上面这段代码看上去合情合理，而实际上却无法通过编译。因为using说明符无法将基类的私有成员函数引入子类当中，即使这里我

们将代码d.foo(5)删除，即不再调用基类的函数，编译器也是不会

让这段代码编译成功的。

## 显式默认和显式删除

为了解决以上种种问题，C++11标准提供了一种方法能够简单有效又精确地控制默认特殊成员函数的添加和删除，我们将这种方法叫作显式默认和显式删除。显式默认和显式删除的语法非常简单，只需要在声明函数的尾部添加`=default`和`=delete`，它们分别指示编译器添加特殊函数的默认版本以及删除指定的函数：

```cpp
struct type {
	type() = default;
	virtual ~type() = delete;
	type(const type &);
};
type::type(const type &) = default;
```

以上代码显式地添加了默认构造和复制构造函数，同时也删除了析构函数。请注意，`=default`可以添加到类内部函数声明，也可以添加到类外部。这里默认构造函数的`=default`就是添加在类内部，而复制构造函数的`=default`则是添加在类外部。提供这种能力的意义在于，**它可以让我们在不修改头文件里函数声明的情况下，改变函数内部的行为**，例如：

```cpp
// type.h
struct type {
	type();
	int x;
};

// type1.cpp
type::type() = default;

// type2.cpp
type::type() { x = 3; }
```

=delete与=default不同，它必须添加在类内部的函数声明中，如果将其添加到类外部，那么会引发编译错误。

通过使用=default，我们可以很容易地解决之前提到的前两个问题，请观察以下代码：

```cpp
class NonTrivial {
	int i;
public:
	Nontrivial (int n) : i(n), j(n) {}
	Nontrivial() {}
	int j;
};

class Trivial {
	int i;
public:
	Trivial(int n) : i(n), j(n) {}
	Trivial() = default;
	int j;
};

int main() {
	Trivial a(5);
	Trivial b;
	b = a;
	std::cout << "std::is_trivial_v<Trivial> : " << std::is_trivial_v<Trivial> << std::endl;
	std::cout << "std::is_trivial_v<NonTrivial> : " << std::is_trivial_v<NonTrivial> << std::endl; 
}
```

注意，我们只是将构造函数`NonTrivial() {}`替换为显式默认构造函数`Trivial() = default`，类就从非平凡类型恢复到平凡类型了。这样一来，既让编译器为类提供了默认构造函数，又保持了类本身的性质，可以说完美解决了之前的问题。

```cpp
class NonCopyable {
public:
	NonCopyable() = default; // 显示添加默认构造函数
	NonCopyable(const NonCopyable&) = delete; // 显示删除复制构造函数
	NonCopyable& operator=(const NonCopyable&) = delete; // 显示删除复制赋值运算符函数
};

int main() {
	NonCopyable a, b;
	a = b; // 编译失败，复制赋值运算符已被删除
}
```

以上代码删除了类`NonCopyable`的复制构造函数和复制赋值运算符函数，这样就禁止了该类对象相互之间的复制操作。请注意，由于显式地删除了复制构造函数，导致默认情况下编译器也不再自动添加默认构造函数，因此我们必须显式地让编译器添加默认构造函数，否则会导致编译失败。

最后，让我们用= delete来解决禁止重载函数的继承问题，这里只需要对基类Base稍作修改即可：

```cpp
class Base {
// void foo(long &);
public:
	void foo(long &) = delete; // 删除 foo(long &) 函数
	void foo(int) {}
};

class Derived : public Base {
public:
	using Base::foo;
	void foo(const char*) {}
};

int main() {
	Derived d;
	d.foo("hello");
	d.foo(5);
}
```

请注意，上面对代码做了两处修改。第一是将`foo(long &)`函数从private移动到public，第二是显式删除该函数。如果只是显式删除了函数，却没有将函数移动到public，那么编译还是会出错的。

## 显式删除的其他用法

显式删除不仅适用于类的成员函数，对于普通函数同样有效。只不过相对于应用于成员函数，应用于普通函数的意义就不大了：

```cpp
void foo() = delete;
static void bar() = delete;
int main() {
	bar(); // 编译失败，函数已经被显式删除
	foo(); // 编译失败，函数已经被显式删除
}
```

另外，显式删除还可以用于类的new运算符和类析构函数。显式删除特定类的new运算符可以阻止该类在堆上动态创建对象，换句话说它可以限制类的使用者只能通过自动变量、静态变量或者全局变量的方式创建对象，例如：

```cpp
struct type {
	void* operator new(std::size_t) = delete;
};

type global_var;
int main() {
	static type static_var;
	type auto_var;
	type* var_ptr = new type; // 编译失败， 该类的 new 已被删除
}
```

显式删除类的析构函数在某种程度上和删除new运算符的目的正好相反，它阻止类通过自动变量、静态变量或者全局变量的方式创建对象，但是却可以通过new运算符创建对象。原因是删除析构函数后，类无法进行析构。所以像自动变量、静态变量或者全局变量这种会隐式调用析构函数的对象就无法创建了，当然了，通过new运算符创建的对象也无法通过delete销毁，例如：

```cpp
struct type {
	~type() = delete;
};
type global_var; // 编译失败，析构函数被删除无法隐式调用

int main() {
	static type static_var; // 编译失败，析构函数被删除无法隐式调用
	type auto_var;          // 编译失败，析构函数被删除无法隐式调用
	type* var_ptr = new type; 
	delete var_ptr;         // 编译失败，析构函数被删除无法显式调用
}
```

通过上面的代码可以看出，只有new创建对象会成功，其他创建和销毁操作都会失败，所以这样的用法并不多见，大部分情况可能在单例模式中出现。

## explicit和=delete

在类的构造函数上同时使用explicit和=delete是一个不明智的做法，它常常会造成代码行为混乱难以理解，应尽量避免这样做。下面这个例子就是反面教材：

```cpp
struct type {
	type(long long) {}
	explicit type(long) = delete;
};
void foo(type) {}

int main() {
	foo(type(58));
	foo(58);
}
```

