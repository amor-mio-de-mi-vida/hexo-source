---
date: 2024-11-24 22:58:54
date modified: 2024-11-25 11:23:50
title: Modern cpp feature 11~20
tags:
  - cpp
categories:
  - cpp
date created: 2024-09-25 13:26:08
---
Reference: 《现代C++语言核心特性解析》

11. 非受限联合类型 (C++ 11)

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