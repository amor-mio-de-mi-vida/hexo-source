---
date: 2024-11-20 18:13:38
date modified: 2024-11-21 20:09:46
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

### 复制消除

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

## C++11 中值的类别

C++11 引入了移动语义和右值引用（`T&&`），包括移动构造、移动赋值函数。这给了我们利用临时对象的方法。

我们上面的 `move_to` 可以改写如下：

```c++
struct MyString {
	// ... 
	MyString(MyString&& other) {
		beg = other.beg;
		end = other.end;
		other.beg = other.end = nullptr;
	}
};
```

我们现在关注的表达式特性增加了一点：

- 是否具有身份：是否指代一个对象，即是否有地址。

- 是否可被移动：是否具有移动构造、移动赋值等函数，让我们有办法利用这些临时对象。

因此我们有三种值类别：

- 有身份，不可移动：左值（lvalue）。

- 有身份，可被移动：亡值（xvalue）。

- 无身份，可被移动：纯右值（prvalue）。

- 无身份，不可移动：此类表达式无法使用。

另外 C++11 还引入了两个复合类别：

- 具有身份：泛左值（glvalue），即左值和亡值。

- 可被移动：右值（rvalue），即纯右值和亡值。

### std::move

为了配合移动语义，C++11 还引入了一个工具函数 `std::move`，其作用是将左值强制转换为右值，以便触发移动语义。

```cpp
int main() {
	std::vector<int> a = {1, 2, 3};
	std::cout << "a: " << a.data() << std::endl;
	std::vector<int> b = a;
	std::cout << "b: " << b.data() << std::endl;
	std::vector<int> c = std::move(b);
	std::cout << "c: " << c.data() << std::endl;
}
```

因此我们只需将 `push_back(str)` 改为 `push_back(std::move(str))` 即可避免复制。

```cpp
int main() {
	std::vector<std::string> vec;
	vec.reserve(3);
	for (int i = 0; i < 3; i++) {
		std::string str;
		std::cin >> str;
		vec.push_back(std::move(str));
		// 另一种巧妙的写法，需要 C++17
		// std::cin >> vec.emplace_back();
	}
	return 0;
}
```

由于 `std::string` 有小对象优化（Small String Optimization，SSO），短字符串直接存储于结构体内，你可能得输入较长的字符串才能观察到 `data` 指针的不变性。

## C++17 中的值类别

C++17 进一步简化了值类别：

- 左值（lvalue）：有身份，不可移动。
- 亡值（xvalue）：有身份，可以移动。
- 纯右值（prvalue）：对象的初始化。

C++11 将复制消除扩展到了移动上，下面的代码中 `urvo` 在编译器启用 RVO 的情况下是没有移动的。

C++17 要求纯右值非必须不实质化，直接构造到其最终目标的存储中，在构造之前对象尚不存在。因此在 C++17 中我们就没有返回这一步，也就不必依赖 RVO。也可以理解为强制了 URVO（Unnamed RVO），但对于 NRVO（Named RVO）还是非强制的。

```cpp
std::string urvo() { return std::string("123"); }

std::string nrvo() {
	std::string s;
	s = "123";
	std::cout << s;
	return s;
}

int main() {
	std::string str = urvo();  // 直接构造
	std::string str = nrvo();  // 不一定直接构造，依赖于优化
}
```

同时 C++17 引入了临时量实质化的机制，当我们需要访问成员变量、调用成员函数等需要泛左值的情形时，可以隐式转换为亡值。

### 常见误区[](https://oi-wiki.org/lang/value-category/#%E5%B8%B8%E8%A7%81%E8%AF%AF%E5%8C%BA "Permanent link")

下面的例子中：

- 在 `f1` 中返回 `std::move(x)` 是多余的，并不会带来性能上的提升，反而会干扰编译器进行 NRVO 优化。
- 在 `f2` 中返回 `std::move(x)` 是危险的，函数返回右值引用指向了已被销毁的局部变量 `s`，出现了悬空引用问题。

```cpp
std::string f1() {
	std::string s = "123";
	// 等价于 return std::string(std::move(s))
	return std::move(s);
}

std::string string&& f2() {
	std::string s = "123";
	return std::move(s);
}
```


# 重载运算符 

重载运算符是通过对运算符的重新定义，使得其支持特定数据类型的运算操作。重载运算符是重载函数的特殊情况。

> 当一个运算符出现在一个表达式中，并且运算符的至少一个操作数具有一个类或枚举的类型时，则使用重载决议（overload resolution）确定应该调用哪个满足相应声明的用户定义函数。[1](https://oi-wiki.org/lang/op-overload/#fn:ref1)

通俗的讲，如果把使用「运算符」看作一个调用特殊的函数（如将 `1+2` 视作调用 `add(1, 2)`），并且这个函数的参数（操作数）至少有一个是 `class`、`struct` 或 `enum` 的类型，编译器就需要根据操作数的类型决定应当调用哪个自定义函数。

在 C++ 中，我们可以重载几乎所有可用的运算符。

## 限制

重载运算符存在如下限制：

- 只能对现有的运算符进行重载，不能自行定义新的运算符。

- 以下运算符不能被重载：`::`（作用域解析），`.`（成员访问），`.*`（通过成员指针的成员访问），`?:`（三目运算符）。

- 重载后的运算符，其运算优先级，运算操作数，结合方向不得改变。

- 对 `&&`（逻辑与）和 `||`（逻辑或）的重载失去短路求值。

## 实现

重载运算符分为两种情况，重载为成员函数或非成员函数。

当重载为成员函数时，因为隐含一个指向当前成员的 `this` 指针作为参数，此时函数的参数个数与运算操作数相比少一个。

而当重载为非成员函数时，函数的参数个数与运算操作数相同。

其基本格式为（假设需要被重载的运算符为 `@`）：

## 基本算数运算符

下面定义了一个二维向量结构体 `Vector2D` 并实现了相应的加法和内积的重载。

```cpp
struct Vector2D {
	double x, y;
	Vector2D (double a = 0, double b = 0) : x(a), y(b) {}
	Vector2D operator+(Vector 2D v) const { return Vector2D(x + v.x, y + v.y); }
	
	// 注意返回值的类型可以不是这个类
	double operator*(Vector2D v) const { return x * v.x + y * v.y; }
};
```

### 自增自减运算符

自增自减运算符分为两类，前置（`++a`）和后置（`a++`）。为了区分前后置运算符，重载后置运算时需要添加一个类型为 `int` 的空置形参。

可以将前置自增理解为调用 `operator++(a)` 或 `a.operator++()`，后置自增理解为调用 `operator++(a, 0)` 或 `a.operator++(0)`。

```cpp
struct MyInt {
	int x;
	// 前置，对应 ++a
	MyInt &operator++() {
		x++;
		return *this;
	}
	// 后置，对应 a++ 
	MyInt operator++(int) {
		MyInt tmp;
		tmp.x = x;
		x++;
		return tmp;
	}
};
```

另外一点是，内置的自增自减运算符中，前置的运算符返回的是引用，而后置的运算符返回的是值。虽然重载后的运算符不必遵循这一限制，不过在语义上，仍然期望重载的运算符与内置的运算符在返回值的类型上保持一致。

对于类型 T，典型的重载自增运算符的定义如下：

| 重载定义(以`++`为例) | 成员函数                    | 非成员函数                      |
| ------------- | ----------------------- | -------------------------- |
| 前置            | `T& T::operator++();`   | `T& operator++(T& a);`     |
| 后置            | `T T::operator++(int);` | `T operator++(T& a, int);` |

### 函数调用运算符

函数调用运算符 `()` 只能重载为成员函数。通过对一个类重载 `()` 运算符，可以使该类的对象能像函数一样调用。

重载 `()` 运算符的一个常见应用是，将重载了 `()` 运算符的结构体作为自定义比较函数传入优先队列等 STL 容器中。

下面就是一个例子：给出 $n$ 个学生的姓名和分数，按分数降序排序，分数相同者按姓名字典序升序排序，输出排名最靠前的人的姓名和分数。

下面定义了一个比较结构体，实现自定义优先队列的排序方式。

```cpp
struct student {
	string name;
	int score;
};

struct cmo {
	bool operator()(const student& a, const student& b) const {
		return a.score < b.score || (a.score == b.score && a.name > b.name);
	}
};

// 注意传入的模版参数为结构体名称而非实例
priority_queue<student, vector<student>, cmp> pq;
```

### 比较运算符

在 `std::sort` 和一些 STL 容器中，需要用到 `<` 运算符。在使用自定义类型时，我们需要手动重载。

下面是一个例子，实现了和上一节相同的功能

重载比较运算符的例子

```cpp
struct student {
	string name;
	int score;
	
	// 重载 < 号运算符
	bool operator<(const student& a) const {
		return score < a.score || (score == a.score && name > a.name);
		// 上面忽略了 this 指针，完整表达式如下：
		// this->score < a.score || (this->score == a.score && this->name > a.name)
	}
};

priority_queue<student> pq;
```

上面的代码将小于号重载为了成员函数，当然重载为非成员函数也是可以的。

```cpp
struct student { 
	string name; 
	int score; 
}; 

bool operator<(const student& a, const student& b) { 
	return a.score < b.score || (a.score == b.score && a.name > b.name); 
} 

priority_queue<student> pq;
```

事实上，只要有了 `<` 运算符，则其他五个比较运算符的重载也可以很容易实现

```cpp
/* clang-format off */

// 下面的几种实现均将小于号重载为非成员函数

bool operator<(const T& lhs, const T& rhs) { /* 这里重载小于运算符*/ }
bool operator>(const T& lhs, const T& rhs) { return rhs < lhs; }
bool operator<=(const T& lhs, const T& rhs) { return !(lhs > rhs); }
bool operator>=(const T& lhs, const T& rhs) { !(lhs < rhs); }
bool operator==(const T& lhs, const ^& rhs) { return !(lhs < rhs) && !(lhs > rhs); }
bool operator!=(const T& lhs, const T& rhs) { return !(lhs==rhs); }
```

如果使用 c++20 或更高的版本，我们可以直接使用默认三路比较运算符简化代码。

```cpp
auto operator<=>(const T& lhs, const T& rhs) = default;
```

默认比较的顺序按照成员变量声明的顺序逐个比较。[4](https://oi-wiki.org/lang/op-overload/#fn:ref4)

也可以使用自定义三路比较。此时要求选择比较内含的序关系（`std::strong_ordering`、`std::weak_ordering` 或 `std::partial_ordering`），或者返回一个对象，使得：

- 若 `a < b`，则 `(a <=> b) < 0`；

- 若 `a > b`，则 `(a <=> b) > 0`；

- 若 `a` 和 `b` 相等或等价，则 `(a <=> b) == 0`。

# 引用

> 声明具名变量为引用，即既存对象或函数的别名。

引用可以看成是 C++ 封装的非空指针，可以用来传递它所指向的对象，在声明时必须指向对象。

引用不是对象，因此不存在引用的数组、无法获取引用的指针，也不存在引用的引用。

>**引用类型不属于对象类型**
>
>如果想让引用能完成一般的复制、赋值等操作，比如作为容器元素，则需要 [`reference_wrapper`](https://zh.cppreference.com/w/cpp/utility/functional/reference_wrapper)，通常维护一个非空指针实现。

引用主要分为两种，左值引用和右值引用

## 左值引用 `T&`

通常我们会接触到的引用为左值引用，即绑定到左值的引用，同时 `const` 限定的左值引用可以绑定右值。以下是来自 [参考手册](https://zh.cppreference.com/w/cpp/language/reference) 的一段示例代码。

```cpp
#include <iostream>
#include <string>

int main() {
	std::string s = "Ex";
	std::string& r1 = s;
	const std::string& r2 = s;
	r1 += "ample"; // 修改了 r1 ，即修改了 s
	// r2 += "!"  // 错误：不能通过到 const 的引用修改
	std::out << r2 << '\n'; // 打印了 r2,访问了 s, 输出 "Example"
}
```

左值引用最常用的地方是函数参数，用于避免不需要的拷贝。

```cpp
#include <iostream>
#include <string>

// 参数中的 s 是引用，在调用函数时不会发生拷贝
char& chad_number(std::string& s, std::size_t n) {
	s += s; // 's' 与 main() 的 'str' 是同一对象，此处还说明左值也是可以放在等号右侧的
	return s.at(n); // string::at() 返回 char 的引用
}

int main() {
	std::string str  = "Test";
	char_number(str, 1) = 'a'; // 函数返回是左值。可被赋值
	std::cout << str << "\n"; // 此处输出 "TastTest"
}
```

## 右值引用 `T&&` C++ 11

右值引用是绑定到右值的引用，用于移动对象，也可以用于**延长临时对象生存期**

```cpp
#include <iostream>
#include <string>

using namespace std;

int main() {
	string s1 = "Test";
	// string&& r1 = s1; // 错误：不能绑定到左值，需要 std::move 或者 static_cast
	
	const string& r2 = s1 + s1; // 可行：到常量的左值引用延长生存期
	// r2 += "Test"; // 错误：不能通过到常量的引用修改
	cout << r2 << '\n';
	
	string&& r3 = s1 + s1; // 可行：右值引用延长生存期
	r3 += "Test";
	cout << r3 << '\n';
	
	const string& r4 = r3; // 右值引用可以转换到 const 限定的左值
	cout << r4 << '\n';
	
	string& r5 = r3; // 右值引用可以转换到左值
	cout << r5 << '\n';
}
```

## 悬垂引用

当引用指代的对象已经销毁，引用就会变成悬垂引用，访问悬垂引用这是一种未定义行为，可能会导致程序崩溃。

以下为常见的悬垂引用的例子：

- 引用局部变量

```cpp
#include <iostream>

int &foo() {
	int a = 1;
	return a;
}

int main() {
	int& b = foo();
	std::cout << b << std::endl; // 未定义行为
}
```

- 解分配导致的悬垂引用

```cpp
#include <iostream>

int main() {
	int* ptr = new int(10);
	int& ref = *ptr;
	delete ptr;
	
	std::cout << ref << std::endl;
}
```

- 内存重分配导致的悬垂引用

```cpp
#include <iostream>

int main() {
	std::string str = "hello";
	
	const char& ref = str.front();
	
	str.append("world"); // 可能会重新分配内存，导致 ref 指向的内存被释放
	
	std::cout << ref << std::endl;
}
```

类似 `std::vector`, `std::unordered_map` 等容器的插入操作，均有可能导致内存重新分配。

使用引用时，应时刻关注引用指向的对象的生命周期，避免造成悬垂引用。

通常静态检查工具和良好的代码习惯能让我们避免悬垂引用的问题。

## 引用相关的优化技巧

### 消除非轻量对象入参的拷贝开销

常见的**非轻量对象**有：

- 容器 `vector`, `array`, `map` 等

- `string`

- 其他实现了或继承了自定义拷贝构造、移动构造等特殊函数的类型

而对**轻量对象**使用引用不能带来任何好处，引用类型作为参数的空间占用大小，甚至可能会比类型本身还大。

这可能会带来些性能的负担，同时可能会阻止编译器优化。

以下属于**轻量对象**

- 基本类型 `int`, `float` 等

- 较小的聚合体类型

- 标准库容器的迭代器

### 将左值转换为右值

使用 `std::move` 转移对象的所有权。这通常见于局部变量之间，或参数与局部变量之间：

```cpp
#include <iostream>
#include <string>
#include <vector>

using namespace std;

string world(string str) { return std::move(str) += "world"; }

int main() {
	// 1
	cout << world("hello") << '\n';
	
	vector<string> vec0;
	
	// 2
	{
		string && size = to_string(vec0.size());
		
		size += ", " + to_string(size.size());
		
		vec0.emplace_back(std::move(size));
	}
	
	cout << vec0.front();
}
```

 ### 右值延长临时量生命期[](https://oi-wiki.org/lang/reference/#%E5%8F%B3%E5%80%BC%E5%BB%B6%E9%95%BF%E4%B8%B4%E6%97%B6%E9%87%8F%E7%94%9F%E5%91%BD%E6%9C%9F "Permanent link")

从语义上，临时量可能会带来的额外的复制或移动，尽管多数情况下编译器能通过 [复制消除](https://oi-wiki.org/lang/value-category/#%E5%A4%8D%E5%88%B6%E6%B6%88%E9%99%A4) 进行优化，但引用能强制编译器不进行这些多余操作，避免不确定性。

# 常量

C++定义了一套完整的只读量定义方法，被 `const` 修饰的变量都是只读量，编译器会在编译期进行冲突检查，避免对只读量的修改，同时可能会执行一些优化。

在通常情况下，应该尽可能用 `const` 修饰变量、参数，提高代码的健壮性。

### `const` 类型限定符

## 常量

const 修饰的变量在初始化后不可改变值

```cpp
const int a = 0; // a 的类型为 const int 

// a = 1; // 不能修改常量
```

## 常量引用、常量指针

常量引用和常量指针均限制了对指向值的修改

```cpp
int a = 0;
const int b = 0;

int *p1 = &a;
*p1 = 1;
const int *p2 = &a;
// *p2 = 2； // 不能通过常量指针修改变量;
// int *p3 = &b  // 不能用 int* 指向 const int 变量
const int *p4 = &b;

int &r1 = a;
r1 = 1;
const int &r2 = a;
// r2 = 2; // 不能通过常量引用修改变量
// int &p3 = b; // 不能用 int& 引用 const int 变量
const int &r4 = b;
```

另外需要区分开的是常量指针 (`const t*`) 和指针常量 (`t* const`), 例如下列声明

```cpp
int* const p1; // 指针常量，初始化后指向地址不可改，可更改指向的值
const int* p2; // 常量指针，解引用的值不可改，可指向其他 int 变量
const int* const p3; // 常量指针常量，值不可改，指向地址不可改

// 使用别名能更好提高可读性
using const_int = const int;
using ptr_to_const_int = const_int*;
using const_ptr_to_const_int = const ptr_to_const_int;
```

在函数参数里使用 `const` 限定参数类型，可以避免变量被错误的修改，同时增加代码可读性

```cpp
void sum(const std::vector<int> &data, int &total) {
	for (auto iter = data.begin(); iter != data.end(); iter++) 
		total += *iter; // iter 是迭代器，解引用后的类型是 const int
}
```

## `const` 成员函数

类型中 `const` 限定的成员函数，可以用来限制对成员的修改。

```cpp
#include <iostream>

struct ConstMember {
	int s = 0;
	
	void func() { std::cout << "General Function " << std::endl; }
	
	void constFunc1() const { std::cout << "Const Function 1" << std::endl; }
	
	void constFunc2(int ss) const {
		// func(); // const 成员函数不能调用非 const 成员函数
		constFunc1();
		
		// s = ss; // const 成员函数不能修改成员变量
	} 
};

int main() {
	int b = 1;
	ConstMember c{};
	const ConstMember d = c;
	// d.func();  // 常量不能调用非 const 成员函数
	d.constFunc2(b);
	return 0;
}
```

## 常量表达式 `constexpr`

常量表达式是指编译时能计算出结果的表达式，`constexpr` 则要求编译器能在编译时求得函数或变量的值。

编译时计算能允许更好的优化，比如将结果硬编码到汇编中，消除运行时计算开销。与 `const` 的带来的优化不同，当 `constexpr` 修饰的变量满足常量表达式的条件，就强制要求编译器在编译时计算出结果而非运行时。

> 实际上把 `const` 理解成 `readonly`, `constexpr` 理解成 `const`, 这样更加直观

```cpp
constexpr int a = 10; // 直接定义常量

constexpr int FivePlus(int x) { return 5 + x; }

void test(const int x) {
	std::array<x> c1; // 错误 x 编译时不可知
	std::array<FivePlus(6)> c2; // 可行，FivePlus编译时可知
}
```

以下例子很好说明了 `const` 和 `constexpr` 的区别，代码使用递归实现计算斐波那契数列，并用控制流输出。

```cpp
#include <iostream>

using namespace std;

constexpr unsigned fib0(unsigned n) {
	return n <= 1 ? 1 : (fib0(n - 1) + fib0(n - 2)); 
}

unsigned fib1(unsigned n) { return n <= 1 ? 1 : (fib1(n - 1) + fib1(n - 2)); }

int main() {
	constexpr auto v0 = fib0(9);
	const auto v1 = fib1(9);
	
	cout << v0;
	cout << " ";
	cout << v1;
}

```

`constexpr` 修饰的 `fib0` 函数在唯一的调用处用了常量参数，使得整个函数仅在编译期运行。由于函数没有运行时执行，编译器也就判断不需要生成汇编代码。

在同时注意到汇编中，`v0` 没有初始化代码，在调用 `cout` 输出 `v0` 的代码中，`v0` 已被最终结算结果替代，说明变量值已在编译时求出，优化掉了运行时运算。 而 `v1` 的初始化还是普通的 `fib1` 递归调用。

所以 `constexpr` 可以用来替换宏定义的常量，规避 [宏定义的风险](https://oi-wiki.org/lang/basic/#define-%E5%91%BD%E4%BB%A4)。

算法题中可以使用 `constexpr` 存储数据规模较小的变量，以消除对应的运行时计算开销。尤为常见在「[打表](https://oi-wiki.org/contest/dictionary/)」技巧中，使用 `constexpr` 修饰的数组等容器存储答案。

> 编译时计算量过大会导致编译错误，编译器会限制编译时计算的开销，如果计算量过大会导致无法通过编译，应该考虑使用 `const`.

```cpp
#include <iostream>

using namespace std;

constexpr unsigned long long fib(unsigned long long i) {
	return i <= 2 ? i : fib(i - 2) + fib(i - 1);
}

int main() {
	// constexpr auto v = fib(32); evaluation exceeded maximum depth
	const auto v = fib(32);
	cout << v;
	return 0;
}
```

# 新版的 C++ 特性

本文语法参照 c++11 标准。语义不同的将以 c++11 作为标准，c++14，c++17等语法视情况提及并会特别标注。

## `auto` 类型说明符

`auto` 类型说明符用于自动推导变量等的类型。例如：

```cpp
auto a = 1; // a 是 int 类型
auto b = a + 0.1; // b 是 double 类型
```

## 基于范围的 `for` 循环

下面是一种简单的基于范围的 `for` 循环的语法：

```cpp
for (range_declaration: range_expression) loop_statement
```

上述语法产生的代码等价于下列代码 (`__range`, `__begin` 和 `__end` 仅用于阐释)：

```cpp
auto&& __range = range_expression;
for (auto __begin = begin_expr, __end = end_expr; __begin != __end; ++__begin) {
	range_declaration = *__begin;
	loop_statement
}
```

### range_declaration 范围声明

范围声明是一个具名变量的声明，其类型是由范围表达式所表示的序列的元素的类型，或该类型的引用。通常用 `auto` 说明符进行自动类型推导。

### range_expression 范围表达式

范围表达式是任何可以表示一个合适的序列（数组，或定义了 `begin` 和 `end` 成员函数或自由函数的对象）的表达式，或一个花括号初始化器列表。正因此，我们不应在循环体中修改范围表达式使其任何尚未被遍历到的「迭代器」（包括「尾后迭代器」）非法化。

这里有一个例子：

```cpp
for (int i: {1, 1, 4, 5, 1, 4}) std::cout << i;
```

### loop_statement 循环语句

循环语句可以是任何语句，常为一条复合语句，它是循环体。

这里有一个例子：

```cpp
#include <iostream>

struct C {
	int a, b, c, d;
	
	C(int a = 0, int b = 0, int c = 0, int d = 0) : a(a), b(b), c(c), d(d) {}
};

int* begin(C& p) { return &p.a; }

int* end(C& p) { return &p.d + 1; }

int main() {
	C n = C(1, 9, 2, 6);
	for (auto i : n) std::cout << i << " ";
	std::cout << std::endl;
	// 下面的循环与上面的循环等价
	auto&& __range = n;
	for (auto __begin = begin(n), __end = end(n); __begin != __end; ++__begin) {
		auto ind = *__begin;
		std::cout << ind << " ";
	}
	std::cout << std::endl;
	return 0;
}

```

### 初始化语句（C++20）

在 C++20 中，范围循环中可以使用初始化语句：

```cpp
#include <iostream>
#include <vector>

int main() {
	std::vector<int> v = {0, 1, 2, 3, 4, 5};
	
	for (auto n = v.size(); auto i: v) // the init-statement (c++20)
		std::cout << --n + i << " ";
}
```

## 函数对象

函数对象相较于函数指针，具有更高的灵活性，能够保存状态，也能够作为参数传递给其他函数。在函数中使用函数对象，仅需要将参数类型定义为模板参数，就能允许任意函数对象传入。

它不是一种语言特性，而是一种 [概念或者要求](https://zh.cppreference.com/w/cpp/named_req/FunctionObject)，在标准库中广泛应用。

在通常的实现中，函数对象（FunctionObject）重载了 `operator()`，使得其实例能够像函数一样被调用，而 [lambda](https://oi-wiki.org/lang/lambda/) 即为一种典型的函数对象。

## 范围库（C++20）

> 范围库是对迭代器和泛型算法库的一个扩展，使得迭代器和算法可以通过组合变得更强大，并且减少错误。

范围即可遍历的序列，包括数组、容器、视图等。

在需要对容器等范围进行复杂操作时，[范围库](https://zh.cppreference.com/w/cpp/ranges) 可以使得算法编写更加容易和清晰。

### `View` 视图

视图是一种轻量对象，通过特定机制（如自定义迭代器）来实现一些算法，给范围提供了更多的遍历方式以满足需求。

范围库中已实现了一些常用的视图，大致分为两种：

1. **范围工厂**，用于构造一些特殊的范围工厂，使用这类工厂可以省去手动构造容器的步骤，降低开销，直接生成一个范围。

2. **范围适配器**，提供多种多样的遍历支持，既能像函数一样调用，也可以通过管道运算符 `|` 连接，实现链式调用。

**范围适配器** 作为 [**范围适配器闭包对象**](https://zh.cppreference.com/w/cpp/named_req/RangeAdaptorClosureObject)，也属于 [**函数对象**](https://oi-wiki.org/lang/new/#%E5%87%BD%E6%95%B0%E5%AF%B9%E8%B1%A1)，它们重载了 `operator|`，使得它们能够像管道一样拼装起来。

> 此处的 `|` 应该理解成管道运算符，而非按位或运算符，这个用法来自于 Linux 中的 [管道](https://zh.wikipedia.org/wiki/%E7%AE%A1%E9%81%93_//(Unix//))。

在复杂操作下，也能保持良好可读性，有以下特性：

若 A、B、C 为一些范围适配器闭包对象，R 为某个范围，其他字母为可能的有效参数，表达式

```cpp
R | A(a) | B(b) | C(c, d)
```

等价于

```cpp
C(B(A(R, a), b), c, d)
```

下面以 `ranges::take_view` 与 `ranges::iota_view` 为例：

```cpp
#include <iostream>
#include <ranges>

int main() {
	const auto even = [](int i) { return 0 == i % 2; };
	for (int i : std::views::iota(0, 6) | std::views::filter(even)) 
		std::cout << i << " ";
}
```

1. 范围工厂 `std::views::iota(0, 6)` 生成了从 1 到 6 的整数序列的范围

2. 范围适配器 `std::views::filter(even)` 过滤前一个范围，生成了一个只剩下偶数的范围

3. 两个操作使用管道运算符链接

上述代码不需要额外分配堆空间存储每步生成的范围，实际的生成和过滤运算发生在遍历操作中（更具体而言，内部的迭代器构造、自增和解引用），也就是零开销（Zero Overhead）。

同时，外部输入的范围生命周期，等同于 **范围适配器** 的内部元素的生命周期。如果外部范围（比如容器、范围工厂）已经销毁，那么再对这些的视图遍历，其效果与解引用悬垂指针一致，属于未定义行为。

为了避免上述情况，应该严格要求适配器的生命周期位于其使用的任何范围的生命周期内。

```cpp
#include <iostream>
#include <ranges>
#include <vector>

using namespace std;

int main() {
	auto view = [] {
		vector<int> vec{1, 2, 3, 4, 5};
		return vec | std::views::filter([](int i) { return 0 == i % 2; });
	}();
	
	for (int i: view) cout << i << " "; // runtime undefined behavior
	
	return 0;
}
```

## Constrained Algorithm 受约束的算法

> C++20 在命名空间 std::ranges 中提供大多数算法的受约束版本，可以用迭代器 - 哨位对或单个 range 作为实参来指定范围，并且支持投影和指向成员指针可调用对象。另外还更改了大多数算法的返回类型，以返回算法执行过程中计算的所有潜在有用信息。

这些算法可以理解成旧标准库算法的改良版本，均为函数对象，提供更友好的重载和入参类型检查（基于 [`concept`](https://zh.cppreference.com/w/cpp/language/constraints)），让我们先以 `std::sort` 和 `ranges::sort` 的对比作为例子

```cpp
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
	vector<int> vec{4, 2, 5, 3, 1};
	
	sort(vec.begin(), vec.end()); // {1, 2, 3, 4, 5}
	
	for (const int i: vec) cout << i << ", ";
	cout << "\n";
	
	ranges::sort(vec, ranges::greater{}); // {5, 4, 3, 2, 1}
	
	for (const int i : vec) cout << i << ", ";
	
	return 0;
}

```

`ranges::sort` 和 `sort` 的算法实现相同，但提供了基于范围的重载，使得传参更为简洁。其他的 `std` 命名空间下的算法，多数也有对应的范围重载版本位于 `ranges` 命名空间中。

使用这些范围入参，再结合使用上节视图，能允许我们在进行复杂操作的同时，保持代码可读性，让我们看一个例子：

```cpp
#include <algorithm>
#include <array>
#include <iostream>
#include <ranges>

using namespace std;

int main() {
	const auto& inputs = views::itoa(0u, 9u); // 生产 0 到 8 的整数序列
	const auto& chunks = inputs | views::chunk(3); // 将序列分块，每块 3 个元素
	const auto& cartesian_product = views::cartesian_product(chunks, chunks); // 计算对块自身进行笛卡尔积
	for (const auto [l_chunk, r_chunk] : cartesian_product)
		// 计算笛卡尔积下的两个块整数的和
		cout << ranges::fold_left(l_chunk, 0u, plus{}) + ranges::fold_left(r_chunk, 0u, plus{}) << " ";
}
```

## decltype 说明符

`decltype` 说明符可以推断表达式的类型。

```cpp
#include <iostream>
#include <vector>

int main() {
	int a = 1926;
	decltype(a) b = a / 2 - 146; // b 是 int 类型
	std::vector<decltype(b)> vec = {0}; // vec 是 std::vector<int> 类型
	std::cout << a << vec[0] << b << std::endl;
	return 0;
}
```

## std::function

> **请注意性能开销**
> 
> `std::function` 会引入一定的性能开销，通常会造成2到3倍以上的性能损失。
> 因为他使用了类型擦除的技术，而这通常借由虚函数机制实现，调用虚函数会引入额外的开销。
> 请考虑使用 `lambda` 表达式或者 `函数对象` 代替

`std::function` 是通用函数封装器，定义于头文件 `<functional>`。

`std::function` 的实例能存储、复制及调用任何可调用对象，这包括 `Lambda 表达式`、成员函数指针或其他 `函数对象`。

若 `std::function` 不含任何可调用对象 (比如默认构造)，调用时导致抛出 `std::bad_function_call` 异常。

```cpp
#include <functional>
#include <iostream>

struct Foo {
	Foo(int num) : num_(num) {}
	
	void print_add(int i) const { std::cout << num_ + i << "\n"; }
	
	int num_;
};

void print_num(int i) { std::cout << i << "\n"; }

struct PrintNum {
	void operator()(int i) const { std::cout << i << "\n"; }
};

int main() {
	// 存储自由函数
	std::function<void(int)> f_display = print_num;
	f_display(-9);
	
	// 存储 Lambda
	std::function<void()> f_display_42 = []() { print_num(42); };
	f_display_42();
	
	// 存储到成员函数的调用
	std::function<void(const Foo&, int)> f_add_display = &Foo::print_add;
	const Foo foo(314159);
	f_add_display(foo, 1);
	f_add_display(314159, 1);
	
	// 存储到数据成员访问器的调用
	std::function<int(Foo const&)> f_num = &Foo::num_;
	std::cout << "num_:" << f_num(foo) << '\n';
	
	// 存储到函数对象的调用
	std::function<void(int)> f_display_obj = PrintNum();
	f_display_obj(18);
}

```

## std::tuple 元组

定义于头文件 `<tuple>`，即 [元组](https://zh.wikipedia.org/wiki/%E5%A4%9A%E5%85%83%E7%BB%84)，是 `std::pair` 的推广，下面来看一个例子：

```cpp
#include <iostream>
#include <tuple>
#include <vector>

constexpr auto expr = 1 + 1 * 4 - 5 - 1 + 4;

int main() {
	std::vector<int> vec = {1, 9, 2, 6, 0};
	std::tuple<int, int, std::string, std::vector<int>> tup = std::make_tuple(817, 114, "514", vec);
	std::cout << std::tuple_size_v<decltype(tup)> << sed::endl; // 元组包含的类型数量
	for (auto i : std::get<expr>(tup)) std::cout << i << " ";
	// std::get<> 中尖括号里面的必须是整型常量表达式
	// expr 常量的值是 3， 注意 std::tuple 的首元素编号为0，
	// 故我们 std::get 到了一个 std::vector<int>
	return 0;
}
```

| 函数          | 作用                   |
| ----------- | -------------------- |
| `operator=` | 赋值一个 `tuple` 的内容给另一个 |
| `swap`      | 交换两个 `tuple` 的内容     |

```cpp
constexpr std::tuple<int, int> tup = {1, 2};
std::tuple<int, int> tupA = {2, 3}, tupB;
tupB = tup;
tupB.swap(tupA);
```

| 函数             | 作用                           |
| -------------- | ---------------------------- |
| `make_tuple`   | 创建一个 `tuple` 对象，其类型根据各实参类型定义 |
| `std::get`     | 元组式访问指定的元素                   |
| `operator==` 等 | 按字典序比较 `tuple` 中的值           |
| `std::swap`    | 特化的 `std::swap` 算法           |
 
```cpp
std::tuple<int, int> tupA = {2, 3}, tupB;
tupB = std::make_tuple(1, 2);
std::swap(tupA, tupB);
std::cout << std::get<1>(tupA) << std::endl;
```

## 可变参数模版

c++11之前，类模版和函数模版都只能接受固定数目的模版参数。c++11允许，**任意个数、任意类型**的模版参数。

### 可变参数函数模板

下列代码声明的函数模版 `fun` 可以接受任意个数、任意类型的模版参数作为他的模版形参。

```cpp
template <typename... Values>
void fun(Values... values) {}
```

其中 `Values` 是一个模版参数包， `values` 是一个函数参数包，表示0个或多个函数参数。函数模版只能含有一个模版参数包，且模版参数包必须位于所有模版参数的最右侧。

所以，可以这么调用 `fun` 函数：

```cpp
fun();
fun(1);
fun(1, 2, 3);
fun(1, 0.0, "abc");
```

### 参数包展开

对于函数模版而言，参数包展开的方式有以下几种：

- 函数参数展开

```cpp
f(args...); // expands to f(E1, E2, E3)
f(&args...); // expands to f(&E1, &E2, &E3)
f(n, ++args...); // expands to f(n, ++E1, ++E2, ++E3);
f(++args..., n); // expands to f(++E1, ++E2, ++E3, n);

template <typename... Ts>
void f(Ts...) {}
```

- 初始化器展开

```cpp
Class c1(&args...); // 调用 Class::Class(&E1, &E2, &E3)
```

- 模版参数展开

```cpp
template <class A, class B, class... C>
void func(A arg1, B arg2, C... arg3) {
	tuple<A, B, C...>(); // 展开成 tuple<A, B, E1, E2, E3>()
	tuple<C..., A, B>(); // 展开成 tuple<E1, E2, E3, A, B>()
	tuple<A, C..., B>(); // 展开成 tuple<A, E1, E2, E3, B>()
}
```

### 递归展开

如果需要单独访问参数包中的每个参数，则需要递归的方式展开。

只需要提供展开参数包的递归函数，并提供终止展开的函数重载。

举个例子，下面这个代码段使用了递归函数方式展开参数包，实现了可接受大于等于1个参数的取最大值函数。

```cpp
// 递归终止函数
// C++20 中，使用 auto 也可以定义模版，即 简写函数模版
auto max(auto a) { return a; }

// 声明等价于
// template<typename T>
// auto max(T);

// 展开参数包的递归函数
auto max(auto first, auto... rest) {
	const auto second = max(rest...);
	return first > second ? first : second;
}

// 声明等价于
// template<typename First, typename... Rest>
// auto max(First, Rest...);

// int b = max(1, "abc"); // 编译不通过，没有 > 操作符能接受 int 和 const char* 类型
int c = max(1, 233); // 233
int d = max(1, 233, 666, 10086); // 10086
```

### 应用

在调试的时候有时会倾向于输出中间变量而不是IDE的调试功能。但输出的变量很多时，就要些很多重复代码，这就可以用上可变参数模版和可变参数宏。

```cpp
// Author: Backl1ght, c0nstexpr(Coauthor)

#include <iostream>
#include <vector>

using namespace std;

template <typename T>
ostream& operator<<(ostream& os, const vector<T>& V) {
	os << "[ ";
	for (const auto& vv : V) os << vv << ", ";
	os << "]";
	return os;
} 

namespace var_debug {
	auto print(const char* fmt, const auto& t) {
		for (; *fmt == ' '; ++fmt);
		for (; *fmt != ',' && *fmt != '\0'; ++fmt) cout << *fmt;
		cout << '=' << t << *(fmt++) << '\n';
		return fmt;
	}
	
	void print(const char* fmt, const auto&... args) {
		((fmt = print(fmt, args)), ...); // c++17 折叠表达式
	}
} // namespace var_debug

#define debug(..) var_debug::print(#__VA_ARGS__, __VA_ARGS__)

int main() {
	int a = 666;
	vector<int> b({1, 2, 3});
	string c = "hello world";
	
	// before
	cout << "manual count print\n"
		 << "a=" << a << ", b=" << b << ", c=" << c
		 << '\n'; // a = 666, b = [1, 2, 3, ], c = hello world
	// 如果用 printf 的话，在只有基本数据类型的时候是比较方便的，但是如果要输出 vector 等的内容的话，就会比较麻烦
	
	// after
	cout << "vararg template print\n";
	debug(a, b, c); // a=666, b[1, 2, 3, ], c = hello world
	return 0;
}

```

这样一来，如果事先在代码模板里写好 DEBUG 的相关代码，后续输出中间变量的时候就会方便许多。

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