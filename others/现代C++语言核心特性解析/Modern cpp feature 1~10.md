---
date: 2024-11-21 20:22:55
date modified: 2024-11-23 20:42:15
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

# decltype 说明符

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
