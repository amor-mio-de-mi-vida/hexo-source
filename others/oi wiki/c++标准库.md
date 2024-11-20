---
date: 2024-11-20 09:01:49
date modified: 2024-11-20 09:19:46
title: c++标准库
tags:
  - oiwiki
categories:
  - oiwiki
---
引用链接：

[OI Wiki - OI Wiki](https://oi-wiki.org/)

[C++ tuple元组的基本用法(总结)-CSDN博客](https://blog.csdn.net/sevenjoin/article/details/88420885)
<!--more-->

# STL容器

## 分类

![Pasted-image-20241119141514](https://github.com/amor-mio-de-mi-vida/picx-images-hosting/raw/master/其他/Pasted-image-20241119141514.4xukj1c2hv.webp)

### 序列式容器

- **向量**(`vector`) 后端可高效增加元素的顺序表。

- **数组**(`array`)**C++11**，定长的顺序表，C 风格数组的简单包装。

- **双端队列**(`deque`) 双端都可高效增加元素的顺序表。

- **列表**(`list`) 可以沿双向遍历的链表。

- **单向列表**(`forward_list`) 只能沿一个方向遍历的链表。

### 关联式容器

- **集合**(`set`) 用以有序地存储 **互异** 元素的容器。其实现是由节点组成的红黑树，每个节点都包含着一个元素，节点之间以某种比较元素大小的谓词进行排列。

- **多重集合**(`multiset`) 用以有序地存储元素的容器。允许存在相等的元素。

- **映射**(`map`) 由 {键，值} 对组成的集合，以某种比较键大小关系的谓词进行排列。

- **多重映射**(`multimap`) 由 {键，值} 对组成的多重集合，亦即允许键有相等情况的映射。

### 无序(关联式)容器

- **无序（多重）集合**(`unordered_set`/`unordered_multiset`)**C++11**，与 `set`/`multiset` 的区别在于元素无序，只关心「元素是否存在」，使用哈希实现。

- **无序（多重）映射**(`unordered_map`/`unordered_multimap`)**C++11**，与 `map`/`multimap` 的区别在于键 (key) 无序，只关心 "键与值的对应关系"，使用哈希实现。

### 容器适配器

容器适配器其实并不是容器。它们不具有容器的某些特点（如：有迭代器、有 `clear()` 函数……）。

- **栈**(`stack`) 后进先出 (LIFO) 的容器，默认是对双端队列（`deque`）的包装。

- **队列**(`queue`) 先进先出 (FIFO) 的容器，默认是对双端队列（`deque`）的包装。

- **优先队列**(`priority_queue`) 元素的次序是由作用于所存储的值对上的某种谓词决定的的一种队列，默认是对向量（`vector`）的包装。

容器都是 `containerName<typeName,...> name` 的形式，但模板参数（`<>` 内的参数）的个数、形式会根据具体容器而变。

### 共有函数

`=`：有赋值运算符以及复制构造函数。

`begin()`：返回指向开头元素的迭代器。

`end()`：返回指向末尾的下一个元素的迭代器。`end()` 不指向某个元素，但它是末尾元素的后继。

`size()`：返回容器内的元素个数。

`max_size()`：返回容器 **理论上** 能存储的最大元素个数。依容器类型和所存储变量的类型而变。

`empty()`：返回容器是否为空。

`swap()`：交换两个容器。

`clear()`：清空容器。

`==`/`!=`/`<`/`>`/`<=`/`>=`：按 **字典序** 比较两个容器的大小。（比较元素大小时 `map` 的每个元素相当于 `set<pair<key, value>>`，无序容器不支持 `<`/`>`/`<=`/`>=`。）


## 序列式容器

### `vector`

`std::vector` 是 STL 提供的 **内存连续的**、**可变长度** 的数组（亦称列表）数据结构。能够提供线性复杂度的插入和删除，以及常数复杂度的随机访问。

#### 构造方法

```c++
// 1. 创建空vector; 常数复杂度 
vector<int> v0; 
// 1+. 这句代码可以使得向vector中插入前3个元素时，保证常数时间复杂度 
v0.reserve(3); 
// 2. 创建一个初始空间为3的vector，其元素的默认值是0; 线性复杂度 
vector<int> v1(3); 
// 3. 创建一个初始空间为3的vector，其元素的默认值是2; 线性复杂度 
vector<int> v2(3, 2); 
// 4. 创建一个初始空间为3的vector，其元素的默认值是1， 并且使用v2的空间配置器; 线性复杂度 
vector<int> v3(3, 1, v2.get_allocator()); 
// 5. 创建一个v2的拷贝vector v4， 其内容元素和v2一样; 线性复杂度 
vector<int> v4(v2); 
// 6. 创建一个v4的拷贝vector v5，其内容是{v4[1], v4[2]}; 线性复杂度 
vector<int> v5(v4.begin() + 1, v4.begin() + 3); 
// 7. 移动v2到新创建的vector v6，不发生拷贝; 常数复杂度; 需要 C++11 
vector<int> v6(std::move(v2)); // 或者 v6 = std::move(v2);
```

#### 元素访问

`vector` 提供了如下几种方法进行元素访问

| 函数           | 作用                                                                |
| ------------ | ----------------------------------------------------------------- |
| `at()`       | `v.at(pos)` 返回容器中下标为 `pos` 的引用。如果数组越界抛出`std::out_of_range` 类型的异常。 |
| `operator[]` | `v[pos]` 返回容器中下标为 `pos` 的引用。不执行越界检查。                              |
| `front()`    | `v.front()` 返回首元素的引用。                                             |
| `back()`     | `v.back()` 返回末尾元素的引用。                                             |
| `data()`     | `v.data()` 返回指向数组第一个元素的指针。                                        |

#### 迭代器

vector 提供了如下几种 迭代器

| 迭代器                  | 作用                                    |
| -------------------- | ------------------------------------- |
| `begin()/cbegin()`   | 返回指向首元素的迭代器，其中 `*begin = front`。      |
| `end()/cend()`       | 返回指向数组尾端占位符的迭代器，注意是没有元素的。             |
| `rbegin()/crbegin()` | 返回指向逆向数组的首元素的逆向迭代器，可以理解为正向容器的末元素。     |
| `rend()/crend()`     | 返回指向逆向数组末元素后一位置的迭代器，对应容器首的前一个位置，没有元素。 |

以上列出的迭代器中，含有字符 `c` 的为只读迭代器，你不能通过只读迭代器去修改 `vector` 中的元素的值。如果一个 `vector` 本身就是只读的，那么它的一般迭代器和只读迭代器完全等价。只读迭代器自 C++11 开始支持。

#### 长度

| 函数           | 作用                                                            |
| ------------ | ------------------------------------------------------------- |
| `empty()`    | 返回一个 `bool` 值，即 `v.begin() == v.end()`，`true` 为空，`false` 为非空。 |
| `size()`     | 返回容器长度（元素数量），即 `std::distance(v.begin(), v.end())`。           |
| `resize()`   | 改变 `vector` 的长度，多退少补。补充元素可以由参数指定。                             |
| `max_size()` | 返回容器的最大可能长度。                                                  |

#### 容量

| 函数                | 作用                                |
| ----------------- | --------------------------------- |
| `reserve()`       | 使得 `vector` 预留一定的内存空间，避免不必要的内存拷贝。 |
| `capacity()`      | 返回容器的容量，即不发生拷贝的情况下容器的长度上限。        |
| `shrink_to_fit()` | 使得 `vector` 的容量与长度一致，多退但不会少。      |

#### 元素增删以及修改

| 函数            | 作用                                                  |
| ------------- | --------------------------------------------------- |
| `clear()`     | 清除所有元素                                              |
| `insert()`    | 支持在某个迭代器位置插入元素、可以插入多个。**复杂度与 `pos` 距离末尾长度成线性而非常数的** |
| `erase()`     | 删除某个迭代器或者区间的元素，返回最后被删除的迭代器。复杂度与 `insert` 一致。        |
| `push_back()` | 在末尾插入一个元素，均摊复杂度为 **常数**，最坏为线性复杂度。                   |
| `pop_back()`  | 删除末尾元素，常数复杂度。                                       |
| `swap()`      | 与另一个容器进行交换，此操作是 **常数复杂度** 而非线性的。                    |

### `array (C++11)`

`std::array` 是 STL 提供的 **内存连续的**、**固定长度** 的数组数据结构。其本质是对原生数组的直接封装。

#### 隐式定义的成员函数

| 函数          | 作用                                  |
| ----------- | ----------------------------------- |
| `operator=` | 以来自另一 `array` 的每个元素重写 `array` 的对应元素 |

#### 元素访问

| 函数           | 作用                   |
| ------------ | -------------------- |
| `at`         | 访问指定的元素，同时进行越界检查     |
| `operator[]` | 访问指定的元素，**不** 进行越界检查 |
| `front`      | 访问第一个元素              |
| `back`       | 访问最后一个元素             |
| `data`       | 返回指向内存中数组第一个元素的指针    |

`at` 若遇 `pos >= size()` 的情况会抛出 `std::out_of_range`。

#### 容量

| 函数         | 作用          |
| ---------- | ----------- |
| `empty`    | 检查容器是否为空    |
| `size`     | 返回容纳的元素数    |
| `max_size` | 返回可容纳的最大元素数 |

由于每个 `array` 都是固定大小容器，`size()` 返回的值等于 `max_size()` 返回的值。

#### 操作

| 函数     | 作用       |
| ------ | -------- |
| `fill` | 以指定值填充容器 |
| `swap` | 交换内容     |

**注意，交换两个 `array` 是 ![](data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7 "\Theta(\text{size})") 的，而非与常规 STL 容器一样为 $O(1)$。**

#### 非成员函数

| 函数             | 作用                  |
| -------------- | ------------------- |
| `operator==` 等 | 按照字典序比较 `array` 中的值 |
| `std::get`     | 访问 `array` 的一个元素    |
| `std::swap`    | 特化的 `std::swap` 算法  |

```c++
// 1. 创建空array，长度为3; 常数复杂度 
std::array<int, 3> v0; 
// 2. 用指定常数创建array; 常数复杂度 
std::array<int, 3> v1{1, 2, 3}; v0.fill(1); // 填充数组 
// 访问数组 
for (int i = 0; i != arr.size(); ++i) cout << arr[i] << " ";
```

### `deque`

`std::deque` 是 STL 提供的 双端队列 数据结构。能够提供线性复杂度的插入和删除，以及常数复杂度的随机访问。

```c++
// 1. 定义一个int类型的空双端队列 v0 
deque<int> v0; 
// 2. 定义一个int类型的双端队列 v1，并设置初始大小为10; 线性复杂度 
deque<int> v1(10); 
// 3. 定义一个int类型的双端队列 v2，并初始化为10个1; 线性复杂度 
deque<int> v2(10, 1); 
// 4. 复制已有的双端队列 v1; 线性复杂度 
deque<int> v3(v1); 
// 5. 创建一个v2的拷贝deque v4，其内容是v4[0]至v4[2]; 线性复杂度 
deque<int> v4(v2.begin(), v2.begin() + 3); 
// 6. 移动v2到新创建的deque v5，不发生拷贝; 常数复杂度; 需要 C++11 
deque<int> v5(std::move(v2));
```

#### 元素访问

与 `vector` 一致，但无法访问底层内存。其高效的元素访问速度可参考实现细节部分。

| 函数           | 作用                                |
| ------------ | --------------------------------- |
| `at()`       | 返回容器中指定位置元素的引用，执行越界检查，**常数复杂度**。  |
| `operator[]` | 返回容器中指定位置元素的引用。不执行越界检查，**常数复杂度**。 |
| `front()`    | 返回首元素的引用。                         |
| `back()`     | 返回末尾元素的引用。                        |

#### 迭代器

与 `vector` 一致。

#### 长度

与 `vector` 一致，但是没有 `reserve()` 和 `capacity()` 函数。（仍然有 `shrink_to_fit()` 函数）

#### 元素增删及修改

与 `vector` 一致，并额外有向队列头部增加元素的函数。

| 函数             | 作用                                                |
| -------------- | ------------------------------------------------- |
| `clear()`      | 清除所有元素                                            |
| `insert()`     | 支持在某个迭代器位置插入元素、可以插入多个。**复杂度与 `pos` 与两端距离较小者成线性**。 |
| `erase()`      | 删除某个迭代器或者区间的元素，返回最后被删除的迭代器。复杂度与 `insert` 一致。      |
| `push_front()` | 在头部插入一个元素，**常数复杂度**。                              |
| `push_back()`  | 在末尾插入一个元素，**常数复杂度**。                              |
| `pop_back()`   | 删除末尾元素，**常数复杂度**。                                 |
| `swap()`       | 与另一个容器进行交换，此操作是 **常数复杂度** 而非线性的。                  |

### `list`

`std::list` 是 STL 提供的 [双向链表](https://oi-wiki.org/ds/linked-list/) 数据结构。能够提供线性复杂度的随机访问，以及常数复杂度的插入和删除。

#### 元素访问

由于 `list` 的实现是链表，因此它不提供随机访问的接口。若需要访问中间元素，则需要使用迭代器。

| 函数        | 作用         |
| --------- | ---------- |
| `front()` | 返回首元素的引用。  |
| `back()`  | 返回末尾元素的引用。 |

#### 操作

`list` 类型还提供了一些针对其特性实现的 STL 算法函数。由于这些算法需要 [随机访问迭代器](https://oi-wiki.org/lang/csl/iterator/)，因此 `list` 提供了特别的实现以便于使用。这些算法有 `splice()`、`remove()`、`sort()`、`unique()`、`merge()` 等。

### `forward_list`

`std::forward_list` 是 STL 提供的 [单向链表](https://oi-wiki.org/ds/linked-list/) 数据结构，相比于 `std::list` 减小了空间开销。

`forward_list` 的使用方法与 `list` 几乎一致，但是迭代器只有单向的

## 关联式容器

### `set`

`set` 是关联容器，含有键值类型对象的已排序集，搜索、移除和插入拥有对数复杂度。`set` 内部通常采用 [红黑树](https://oi-wiki.org/ds/rbtree/) 实现。[平衡二叉树](https://oi-wiki.org/ds/bst/) 的特性使得 `set` 非常适合处理需要同时兼顾查找、插入与删除的情况。

和数学中的集合相似，`set` 中不会出现值相同的元素。如果需要有相同元素的集合，需要使用 `multiset`。`multiset` 的使用方法与 `set` 的使用方法基本相同。

#### 插入与删除操作

| 函数                  | 作用                               |
| ------------------- | -------------------------------- |
| `insert(x)`         | 当容器中没有等价元素的时候，将元素 x 插入到 `set` 中。 |
| `erase(x)`          | 删除值为 x 的 **所有** 元素，返回删除元素的个数。    |
| `erase(pos)`        | 删除迭代器为 pos 的元素，要求迭代器必须合法。        |
| `erase(first,last)` | 删除迭代器在 $[first,last)$ 范围内的所有元素。  |
| `clear()`           | 清空 `set`。                        |

insert 函数的返回值类型为 `pair<iterator, bool>`，其中 iterator 是一个指向所插入元素（或者是指向等于所插入值的原本就在容器中的元素）的迭代器，而 bool 则代表元素是否插入成功，由于 `set` 中的元素具有唯一性质，所以如果在 `set` 中已有等值元素，则插入会失败，返回 false，否则插入成功，返回 true；`map` 中的 insert 也是如此。

#### 迭代器

`set` 提供了以下几种迭代器：

| 迭代器                  | 作用                                    |
| -------------------- | ------------------------------------- |
| `begin()/cbegin()`   | 返回指向首元素的迭代器，其中 `*begin = front`。      |
| `end()/cend()`       | 返回指向数组尾端占位符的迭代器，注意是没有元素的。             |
| `rbegin()/crbegin()` | 返回指向逆向数组的首元素的逆向迭代器，可以理解为正向容器的末元素。     |
| `rend()/crend()`     | 返回指向逆向数组末元素后一位置的迭代器，对应容器首的前一个位置，没有元素。 |

以上列出的迭代器中，含有字符 `c` 的为只读迭代器，你不能通过只读迭代器去修改 `set` 中的元素的值。如果一个 `set` 本身就是只读的，那么它的一般迭代器和只读迭代器完全等价。只读迭代器自 C++11 开始支持。

#### 查找操作

| 函数               | 作用                                           |
| ---------------- | -------------------------------------------- |
| `count(x)`       | 返回 `set` 内键为 x 的元素数量。                        |
| `find(x)`        | 在 `set` 内存在键为 x 的元素时会返回该元素的迭代器，否则返回 `end()`。 |
| `lower_bound(x)` | 返回指向首个不小于给定键的元素的迭代器。如果不存在这样的元素，返回 `end()`。   |
| `upper_bound(x)` | 返回指向首个大于给定键的元素的迭代器。如果不存在这样的元素，返回 `end()`。    |
| `empty()`        | 返回容器是否为空。                                    |
| `size()`         | 返回容器内元素个数。                                   |

> `lower_bound` 和 `upper_bound` 的时间复杂度
> 
> `set` 自带的 `lower_bound` 和 `upper_bound` 的时间复杂度为 $O(\log n)$。
> 
> 但使用 `algorithm` 库中的 `lower_bound` 和 `upper_bound` 函数对 `set` 中的元素进行查询，时间复杂度为 $O(n)$。

> `nth_element` 的时间复杂度
> 
> `set` 没有提供自带的 `nth_element`。使用 `algorithm` 库中的 `nth_element` 查找第 ![](data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7 "k") 大的元素时间复杂度为 $O(n)$。
> 
> 如果需要实现平衡二叉树所具备的 $O(\log n)$ 查找第 ![](data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7 "k") 大元素的功能，需要自己手写平衡二叉树或权值线段树，或者选择使用 `pb_ds` 库中的平衡二叉树。

#### 使用样例

在贪心算法中经常会需要出现类似 **找出并删除最小的大于等于某个值的元素**。这种操作能轻松地通过 `set` 来完成。

```c++
// 现存可用的元素 
set<int> available; 
// 需要大于等于的值 
int x; 

// 查找最小的大于等于x的元素 
set<int>::iterator it = available.lower_bound(x); 
if (it == available.end()) { 
	// 不存在这样的元素，则进行相应操作…… 
	} else { 
	// 找到了这样的元素，将其从现存可用元素中移除 
	available.erase(it); 
	// 进行相应操作…… 
}
```

### `map`

`map` 是有序键值对容器，它的元素的键是**唯一**的。搜索、移除和插入操作拥有对数复杂度。`map` 通常实现为 [红黑树](https://oi-wiki.org/ds/rbtree/)。

设想如下场景：现在需要存储一些键值对，例如存储学生姓名对应的分数：`Tom 0`，`Bob 100`，`Alan 100`。但是由于数组下标只能为非负整数，所以无法用姓名作为下标来存储，这个时候最简单的办法就是使用 STL 中的 `map`。

`map` 重载了 `operator[]`，可以用任意定义了 `operator <` 的类型作为下标（在 `map` 中叫做 `key`，也就是索引）：

```c++
map<Key, T> yourMap;
```

`map` 中不会存在键相同的元素，`multimap` 中允许多个元素拥有同一键。`multimap` 的使用方法与 `map` 的使用方法基本相同。

> 正是因为 `multimap` 允许多个元素拥有同一键的特点，`multimap` 并没有提供给出键访问其对应值的方法。

#### 插入与删除操作

| 函数                    | 作用                                                                                             |
| --------------------- | ---------------------------------------------------------------------------------------------- |
| `operator[]`          | 可以直接通过下标访问来进行查询或插入操作。例如 `mp["Alan"]=100`。                                                      |
| `insert(pair<Key,T>)` | 通过向 `map` 中插入一个类型为 `pair<Key, T>` 的值可以达到插入元素的目的，例如 `mp.insert(pair<string,int>("Alan",100));`； |
| `erase(key)`          | 函数会删除键为 `key` 的 **所有** 元素。返回值为删除元素的数量。                                                         |
| `erase(pos)`          | 删除迭代器为 pos 的元素，要求迭代器必须合法。                                                                      |
| `erase(first,last)`   | 删除迭代器在 $[first,last)$ 范围内的所有元素。                                                                |
| `clear()`             | 函数会清空整个容器。                                                                                     |

> 在利用下标访问 `map` 中的某个元素时，如果 `map` 中不存在相应键的元素，会自动在 `map` 中插入一个新元素，并将其值设置为默认值（对于整数，值为零；对于有默认构造函数的类型，会调用默认构造函数进行初始化）。
> 
> 当下标访问操作过于频繁时，容器中会出现大量无意义元素，影响 `map` 的效率。因此一般情况下推荐使用 `find()` 函数来寻找特定键的元素。

#### 查询操作

| 函数               | 作用                                                            |
| ---------------- | ------------------------------------------------------------- |
| `count(x)`       | 返回容器内键为 x 的元素数量。复杂度为 $O(\log(size)+ans)$（关于容器大小对数复杂度，加上匹配个数）。 |
| `find(x)`        | 若容器内存在键为 x 的元素，会返回该元素的迭代器；否则返回 `end()`。                       |
| `lower_bound(x)` | 返回指向首个不小于给定键的元素的迭代器。                                          |
| `upper_bound(x)` | 返回指向首个大于给定键的元素的迭代器。若容器内所有元素均小于或等于给定键，返回 `end()`。              |
| `empty()`        | 返回容器是否为空。                                                     |
| `size()`         | 返回容器内元素个数。                                                    |

#### 使用样例

在搜索中，我们有时需要存储一些较为复杂的状态（如坐标，无法离散化的数值，字符串等）以及与之有关的答案（如到达此状态的最小步数）。`map` 可以用来实现此功能。其中的键是状态，而值是与之相关的答案。下面的示例展示了如何使用 `map` 存储以 `string` 表示的状态。

```c++
// 存储状态与对应的答案 
map<string, int> record; 

// 新搜索到的状态与对应答案 
string status; int ans; 
// 查找对应的状态是否出现过 
map<string, int>::iterator it = record.find(status); 
if (it == record.end()) { 
	// 尚未搜索过该状态，将其加入状态记录中 
	record[status] = ans; 
	// 进行相应操作…… 
	} else { 
	// 已经搜索过该状态，进行相应操作…… 
}
```

#### 遍历容器

可以利用迭代器来遍历关联式容器的所有元素。

```c++
set<int> s;
using si = set<int>::iterator;
for (si it = s.begin(); it != s.end(); it++) 
	cout << *it << endl;
```

需要注意的是，对 `map` 的迭代器解引用后，得到的是类型为 `pair<Key, T>` 的键值对。

在 C++11 中，使用范围 for 循环会让代码简洁很多：

```c++
set<int> s; for (auto x : s) cout << x << endl;
```

对于任意关联式容器，使用迭代器遍历容器的时间复杂度均为 $O(\log n)$

#### 自定义比较方式

`set` 在默认情况下的比较函数为 `<`（如果是非内置类型需要 [重载 `<` 运算符](https://oi-wiki.org/lang/op-overload/#%E6%AF%94%E8%BE%83%E8%BF%90%E7%AE%97%E7%AC%A6)）。然而在某些特殊情况下，我们希望能自定义 `set` 内部的比较方式。

这时候可以通过传入自定义比较器来解决问题。

具体来说，我们需要定义一个类，并在这个类中 [重载 `()` 运算符](https://oi-wiki.org/lang/op-overload/#%E5%87%BD%E6%95%B0%E8%B0%83%E7%94%A8%E8%BF%90%E7%AE%97%E7%AC%A6)。

例如，我们想要维护一个存储整数，且较大值靠前的 `set`，可以这样实现：

```c++
struct cmp {
	bool operator()(int a, int b) const { return a > b; }
};

set<int, cmp> s;
```

对于其他关联式容器，可以用类似的方式实现自定义比较，这里不再赘述。

## 无序关联式容器

自 C++11 标准起，四种基于 [哈希](https://oi-wiki.org/ds/hash/) 实现的无序关联式容器正式纳入了 C++ 的标准模板库中，分别是：`unordered_set`，`unordered_multiset`，`unordered_map`，`unordered_multimap`。

它们与相应的关联式容器在功能，函数等方面有诸多共同点，而最大的不同点则体现在普通的关联式容器一般采用红黑树实现，内部元素按特定顺序进行排序；而这几种无序关联式容器则采用哈希方式存储元素，内部元素不以任何特定顺序进行排序，所以访问无序关联式容器中的元素时，访问顺序也没有任何保证。

采用哈希存储的特点使得无序关联式容器 **在平均情况下** 大多数操作（包括查找，插入，删除）都能在常数时间复杂度内完成，相较于关联式容器与容器大小成对数的时间复杂度更加优秀。

> 在最坏情况下，对无序关联式容器进行插入、删除、查找等操作的时间复杂度会 **与容器大小成线性关系**！这一情况往往在容器内出现大量哈希冲突时产生。
> 
> 同时，由于无序关联式容器的操作时通常存在较大的常数，其效率有时并不比普通的关联式容器好太多。
> 
> 因此应谨慎使用无序关联式容器，尽量避免滥用（例如懒得离散化，直接将 `unordered_map<int, int>` 当作空间无限的普通数组使用）。

由于无序关联式容器与相应的关联式容器在用途和操作中有很多共同点，这里不再介绍无序关联式容器的各种操作，这些内容读者可以参考 [关联式容器](https://oi-wiki.org/lang/csl/associative-container/)。

### 制造哈希冲突

上文中提到了，在最坏情况下，对无序关联式容器进行一些操作的时间复杂度会与容器大小成线性关系。

在哈希函数确定的情况下，可以构造出数据使得容器内产生大量哈希冲突，导致复杂度达到上界。

在标准库实现里，每个元素的散列值是将值对一个质数取模得到的，更具体地说，是 [这个列表](https://github.com/gcc-mirror/gcc/blob/releases/gcc-8.1.0/libstdc%2B%2B-v3/src/shared/hashtable-aux.cc) 中的质数（g++ 6 及以前版本的编译器，这个质数一般是 $126271$，g++ 7 及之后版本的编译器，这个质数一般是$107897$。

因此可以通过向容器中插入这些模数的倍数来达到制造大量哈希冲突的目的。

### 自定义哈希函数

使用自定义哈希函数可以有效避免构造数据产生的大量哈希冲突。要想使用自定义哈希函数，需要定义一个结构体，并在结构体中重载 `()` 运算符，像这样：

```c++
struct my_hash {
	size_t operator()(int x) const {return x;}
}
```

当然，为了确保哈希函数不会被迅速破解（例如 Codeforces 中对使用无序关联式容器的提交进行 hack），可以试着在哈希函数中加入一些随机化函数（如时间）来增加破解的难度。

例如，在 [这篇博客](https://codeforces.com/blog/entry/62393) 中给出了如下哈希函数：

```c++
struct my_hash { 
	static uint64_t splitmix64(uint64_t x) { 
		x += 0x9e3779b97f4a7c15; 
		x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
		x = (x ^ (x >> 27)) * 0x94d049bb133111eb; 
		return x ^ (x >> 31); 
	} 
	
	size_t operator()(uint64_t x) const { 
		static const uint64_t FIXED_RANDOM = 
				chrono::steady_clock::now().time_since_epoch().count();
		 return splitmix64(x + FIXED_RANDOM); 
	} 
	
	// 针对 std::pair<int, int> 作为主键类型的哈希函数 
	size_t operator()(pair<uint64_t, uint64_t> x) const { 
		static const uint64_t FIXED_RANDOM = 
				chrono::steady_clock::now().time_since_epoch().count(); 
		return splitmix64(x.first + FIXED_RANDOM) ^ 
				(splitmix64(x.second + FIXED_RANDOM) >> 1); 
	} 
};
```

写完自定义的哈希函数后，就可以通过 `unordered_map<int, int, my_hash> my_map;` 或者 `unordered_map<pair<int, int>, int, my_hash> my_pair_map;` 的定义方式将自定义的哈希函数传入容器了。

## 容器适配器

### `stack`

STL [栈](https://oi-wiki.org/ds/stack/)(`std::stack`) 是一种后进先出 (Last In, First Out) 的容器适配器，仅支持查询或删除最后一个加入的元素（栈顶元素），不支持随机访问，且为了保证数据的严格有序性，不支持迭代器。

#### 定义

```c++
std::stack<TypeName> s; // 使用默认底层容器 deque，数据类型为 TypeName 
std::stack<TypeName, Container> s; // 使用 Container 作为底层容器 
std::stack<TypeName> s2(s1); // 将 s1 复制一份用于构造 s2
```

#### 成员函数

**以下所有函数均为常数复杂度**

| 函数        | 作用                  |
| --------- | ------------------- |
| `top()`   | 访问栈顶元素（如果栈为空，此处会出错） |
| `push(x)` | 向栈中插入元素 x           |
| `pop()`   | 删除栈顶元素              |
| `size()`  | 查询容器中的元素数量          |
| `empty()` | 询问容器是否为空            |

#### 简单示例

```c++
std::stack<int> s1; 
s1.push(2); 
s1.push(1); 
std::stack<int> s2(s1); 
s1.pop(); 
std::cout << s1.size() << " " << s2.size() << std::endl; // 1 2 
std::cout << s1.top() << " " << s2.top() << std::endl; // 2 1 
s1.pop(); 
std::cout << s1.empty() << " " << s2.empty() << std::endl; // 1 0
```

### `queue`

STL [队列](https://oi-wiki.org/ds/queue/)(`std::queue`) 是一种先进先出 (First In, First Out) 的容器适配器，仅支持查询或删除第一个加入的元素（队首元素），不支持随机访问，且为了保证数据的严格有序性，不支持迭代器。

#### 定义

```c++
std::queue<TypeName> q; // 使用默认底层容器 deque，数据类型为 TypeName 
std::queue<TypeName, Container> q; // 使用 Container 作为底层容器 

std::queue<TypeName> q2(q1); // 将 s1 复制一份用于构造 q2
```

#### 成员函数

| 函数        | 作用                   |
| --------- | -------------------- |
| `front()` | 访问队首元素（如果队列为空，此处会出错） |
| `push(x)` | 向队列中插入元素 x           |
| `pop()`   | 删除队首元素               |
| `size()`  | 查询容器中的元素数量           |
| `empty()` | 询问容器是否为空             |

#### 简单示例

```c++
std::queue<int> q1; 
q1.push(2); 
q1.push(1); 
std::queue<int> q2(q1); 
q1.pop(); std::cout << q1.size() << " " << q2.size() << std::endl; // 1 2 
std::cout << q1.front() << " " << q2.front() << std::endl; // 1 2 
q1.pop(); 
std::cout << q1.empty() << " " << q2.empty() << std::endl; // 1 0
```

### `priority_queue`

优先队列 `std::priority_queue` 是一种 [堆](https://oi-wiki.org/ds/heap/)，一般为 [二叉堆](https://oi-wiki.org/ds/binary-heap/)。

#### 定义

```c++
std::priority_queue<TypeName> q; // 数据类型为 TypeName 
std::priority_queue<TypeName, Container> q; // 使用 Container 作为底层容器 
std::priority_queue<TypeName, Container, Compare> q; 
// 使用 Container 作为底层容器，使用 Compare 作为比较类型 

// 默认使用底层容器 vector 
// 比较类型 less<TypeName>（此时为它的 top() 返回为最大值） 
// 若希望 top() 返回最小值，可令比较类型为 greater<TypeName> 
// 注意：不可跳过 Container 直接传入 Compare 

// 从 C++11 开始，如果使用 lambda 函数自定义 Compare 
// 则需要将其作为构造函数的参数代入，如： 
auto cmp = [](const std::pair<int, int> &l, const std::pair<int, int> &r) { 
	return l.second < r.second; 
}; 

std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, decltype(cmp)> 
pq(cmp);
```

#### 成员函数

**以下所有函数均为常数复杂度**

| 函数        | 作用                 |
| --------- | ------------------ |
| `top()`   | 访问堆顶元素（此时优先队列不能为空） |
| `empty()` | 询问容器是否为空           |
| `size()`  | 查询容器中的元素数量         |

**以下所有函数均为对数复杂度**

| 函数        | 作用                 |
| --------- | ------------------ |
| `push(x)` | 插入元素，并对底层容器排序      |
| `pop()`   | 删除堆顶元素（此时优先队列不能为空） |

#### 简单示例

```c++
std::priority_queue<int> q1; 
std::priority_queue<int, std::vector<int>> q2; 
// C++11 后空格可省略 
std::priority_queue<int, std::deque<int>, std::greater<int>> q3; 
// q3 为小根堆 
for (int i = 1; i <= 5; i++) q1.push(i); 
// q1 中元素 : [1, 2, 3, 4, 5] 
std::cout << q1.top() << std::endl; 
// 输出结果 : 5 
q1.pop(); 
// 堆中元素 : [1, 2, 3, 4] 
std::cout << q1.size() << std::endl; 
// 输出结果 ：4 
for (int i = 1; i <= 5; i++) q3.push(i); 
// q3 中元素 : [1, 2, 3, 4, 5] 
std::cout << q3.top() << std::endl; 
// 输出结果 : 1
```

# STL算法

## 排序相关STL

### `qsort`

该函数为 C 标准库实现的 [快速排序](https://oi-wiki.org/basic/quick-sort/)，定义在 `<stdlib.h>` 中。在 C++ 标准库里，该函数定义在 `<cstdlib>` 中。

#### 函数原型

```c++
qsort(void* _Base, size_t _NumOfElements, size_t _SizeOfElements, int(* _PtFuncCompare)(const void*, const void*));
```
- `_Base`: 数组名
- `_NumOfElements`: 元素的个数
- `_SizeOfElements`: 元素的大小
- `_PtFuncCompare`: 比较函数

qsort 函数有四个参数：数组名、元素个数、元素大小、比较规则。其中，比较规则通过指定比较函数来实现，指定不同的比较函数可以实现不同的排序规则。

比较函数的参数限定为两个 const void 类型的指针。返回值规定为正数、负数和 0。

比较函数的一种示例写法为：

```c++
int compare(const void *p1, const void* p2) {
	int* a = (int*)p1;
	int* b = (int*)p2;
	if (*a > *b) 
		return 1;
	else if (*a < *b) 
		return -1;
	else
		return 0;
}
```

注意：返回值用两个元素相减代替正负数是一种典型的错误写法，因为这样可能会导致溢出错误。

以下是排序结构体的一个示例：

```c++
struct eg {
	int e;
	int g;
};

int compare(const void* p1, const void* p2) {
	struct eg* a = (struct eg*)p1;
	struct eg* b = (struct eg*)p2;
	if (a->e > b->e) 
		return 1;
	else if (a->e < b->e)
		return -1;
	else 
		return 0;
}
```

这里也可以看出，等价不代表相等，只代表在此比较规则下两元素等价。

### `std::sort`

#### 函数用法

```c++
// a[0] ... a[n-1] 为需要排序的数列
// 对 a 原地排序，将其按从小到大的顺序排列
std::sort(a, a+n);

// cmp 为自定义的比较函数
std::sort(a, a+n, cmp);
```

注意：sort 的比较函数的返回值是 true 和 false，用 true 和 false 表示两个元素的大小（先后）关系，这与 qsort 的三值比较函数的语义完全不同。具体内容详见上方给出的 sort 的文档。

如果要将 sort 简单改写为 qsort，维持排序顺序整体上不变（不考虑等价的元素），需要将返回 true 改为 - 1，返回 false 改为 1。

`std::sort` 函数是更常用的 C++ 库比较函数。该函数的最后一个参数为二元比较函数，未指定 `cmp` 函数时，默认按从小到大的顺序排序。

旧版 C++ 标准中仅要求它的 **平均** 时间复杂度达到 $O(n\log n)$。C++11 标准以及后续标准要求它的 **最坏** 时间复杂度达到 $O(n\log n)$。

C++ 标准并未严格要求此函数的实现算法，具体实现取决于编译器。[libstdc++](https://github.com/mirrors/gcc/blob/master/libstdc++-v3/include/bits/stl_algo.h) 和 [libc++](http://llvm.org/svn/llvm-project/libcxx/trunk/include/algorithm) 中的实现都使用了 [内省排序](https://oi-wiki.org/basic/quick-sort/#%E5%86%85%E7%9C%81%E6%8E%92%E5%BA%8F)。

### `std::nth_element`

#### 函数用法

```c++
std::nth_element(first, nth, last);
std::nth_element(first, nth, last, cmp);
```

它重排 `[first, last)` 中的元素，使得 `nth` 所指向的元素被更改为 `[first, last)` 排好序后该位置会出现的元素。这个新的 `nth` 元素前的所有元素小于或等于新的 `nth` 元素后的所有元素。

实现算法是未完成的内省排序。

对于以上两种用法，C++ 标准要求它的平均时间复杂度为 ![](data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7 "O(n)")，其中 n 为 `std::distance(first, last)`。

它常用于构建 [K-D Tree](https://oi-wiki.org/ds/kdt/)。

### `std::stable_sort`

#### 函数用法

```c++
std::stable_sort(first, last);
std::stable_sort(first, last, cmp);
```

稳定排序，保证相等元素排序后的相对位置与原序列相同。

时间复杂度为 $O(n\log^2 n)$，当额外内存可用时，复杂度为 $O(n\log n)$。

### `std::partial_sort`

#### 函数用法

```c++
// mid = first + k 
std::partial_sort(first, mid, last); 
std::partial_sort(first, mid, last, cmp);
```

将序列中前 `k` 元素按 `cmp` 给定的顺序进行原地排序，后面的元素不保证顺序。未指定 `cmp` 函数时，默认按从小到大的顺序排序。

复杂度：约 $(\mathit{last}-\mathit{first})\log(\mathit{mid}-\mathit{first})$ 次应用 `cmp`。

原理：

`std::partial_sort` 的思想是：对原始容器内区间为 `[first, mid)` 的元素执行 `make_heap()` 操作，构造一个大根堆，然后将 `[mid, last)` 中的每个元素和 `first` 进行比较，保证 `first` 内的元素为堆内的最大值。如果小于该最大值，则互换元素位置，并对 `[first, mid)` 内的元素进行调整，使其保持最大堆序。比较完之后，再对 `[first, mid)` 内的元素做一次堆排序 `sort_heap()` 操作，使其按增序排列。注意，堆序和增序是不同的。

### 自定义比较

参见：[运算符重载](https://zh.cppreference.com/w/cpp/language/operators)

内置类型（如 `int`）和用户定义的结构体允许定制调用 STL 排序函数时使用的比较函数。可以在调用该函数时，在最后一个参数中传入一个实现二元比较的函数。

对于用户定义的结构体，对其使用 STL 排序函数前必须定义至少一种关系运算符，或是在使用函数时提供二元比较函数。通常推荐定义 `operator<`。

示例

```c++
int a[1009], n=10;
// ...
std::sort(a + 1, a + 1 + n); // 从小到大排序
std::sort(a + 1，a + 1 + n, greater<int>()); // 从大到小排序
```

```c++
struct data {
	int a, b;
	
	bool operator<(const data rhs) {
		return (a == rhs.a) ? (b < rhs.b) : (a < rhs.a);
	}
} da[1009];

bool cmp(const data u1, const data u2) {
	return (u1.a == u2.a) ? (u1.b > u2.b) : (u1.a > u2.a);
}

// ...
std::sort(da + 1, da + 1 + 10)； // 使用结构体中定义的 < 运算符，从小到大排序
std::sort(da + 1, da + 1 + 10, cmp);  // 使用 cmp 函数进行比较，从大到小排序
```

## 其他STL算法

| 函数               | 作用                                                                                                                          |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `find`           | 顺序查找。`find(v.begin(), v.end(), value)`，其中 `value` 为需要查找的值。                                                                  |
| `reverse`        | 翻转数组、字符串。`reverse(v.begin(), v.end())` 或 `reverse(a + begin, a + end)`。                                                     |
| `unique`         | 去除容器中相邻的重复元素。`unique(ForwardIterator first, ForwardIterator last)`，返回值为指向 **去重后** 容器结尾的迭代器，原容器大小不变。与 `sort` 结合使用可以实现完整容器去重。 |
| `random_shuffle` | 随机地打乱数组。`random_shuffle(v.begin(), v.end())` 或 `random_shuffle(v + begin, v + end)`。                                        |
> `random_shuffle` 自 C++14 起被弃用，C++17 起被移除。
> 
> 在 C++11 以及更新的标准中，您可以使用 `shuffle` 函数代替原来的 `random_shuffle`。使用方法为 `shuffle(v.begin(), v.end(), rng)`（最后一个参数传入的是使用的随机数生成器，一般情况使用以真随机数生成器 [`random_device`](https://zh.cppreference.com/w/cpp/numeric/random/random_device) 播种的梅森旋转伪随机数生成器 [`mt19937`](https://zh.cppreference.com/w/cpp/numeric/random/mersenne_twister_engine)）。

```c++
// #include <random> 
std::mt19937 rng(std::random_device{}()); 
std::shuffle(v.begin(), v.end(), rng);
```

| 函数              | 作用                                                                                                                                                                                                                                                      |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sort`          | 排序。`sort(v.begin(), v.end(), cmp)` 或 `sort(a + begin, a + end, cmp)`，其中 `end` 是排序的数组最后一个元素的后一位，`cmp` 为自定义的比较函数。                                                                                                                                         |
| `stable_sort`   | 稳定排序，用法同 `sort()`。                                                                                                                                                                                                                                      |
| `nth_element`   | 按指定范围进行分类，即找出序列中第 ![](data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7 "n") 大的元素，使其左边均为小于它的数，右边均为大于它的数。`nth_element(v.begin(), v.begin() + mid, v.end(), cmp)` 或 `nth_element(a + begin, a + begin + mid, a + end, cmp)`。 |
| `binary_search` | 二分查找。`binary_search(v.begin(), v.end(), value)`，其中 `value` 为需要查找的值。                                                                                                                                                                                     |
| `merge`         | 将两个（已排序的）序列 **有序合并** 到第三个序列的 **插入迭代器** 上。`merge(v1.begin(), v1.end(), v2.begin(), v2.end() ,back_inserter(v3))`。                                                                                                                                        |
| `lower_bound`   | 在一个有序序列中进行二分查找，返回指向第一个 **大于等于** ![](data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7 "x") 的元素的位置的迭代器。如果不存在这样的元素，则返回尾迭代器。`lower_bound(v.begin(),v.end(),x)`。                                                               |
| `upper_bound`   | 在一个有序序列中进行二分查找，返回指向第一个 **大于** ![](data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7 "x") 的元素的位置的迭代器。如果不存在这样的元素，则返回尾迭代器。`upper_bound(v.begin(),v.end(),x)`。                                                                 |
> `lower_bound` 和 `upper_bound` 的时间复杂度
> 
> 在一般的数组里，这两个函数的时间复杂度均为 $O(\log n)$，但在 `set` 等关联式容器中，直接调用 `lower_bound(s.begin(),s.end(),val)` 的时间复杂度是 $O(n)$ 的。
> 
> `set` 等关联式容器中已经封装了 `lower_bound` 等函数（像 `s.lower_bound(val)` 这样），这样调用的时间复杂度是 $O(\log n)$ 的。

| 函数                 | 作用                                                                                                                                                                                                            |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `next_permutation` | 将当前排列更改为 **全排列中的下一个排列**。如果当前排列已经是 **全排列中的最后一个排列**（元素完全从大到小排列），函数返回 `false` 并将排列更改为 **全排列中的第一个排列**（元素完全从小到大排列）；否则，函数返回 `true`。`next_permutation(v.begin(), v.end())` 或 `next_permutation(v + begin, v + end)`。 |
| `prev_permutation` | 将当前排列更改为 **全排列中的上一个排列**。用法同 `next_permutation`。                                                                                                                                                               |
| `partial_sum`      | 求前缀和。设源容器为 ![](data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7 "x")，目标容器为 $y$，则令$y[i]=x[0]+x[1]+\dots+x[i]$。`partial_sum(src.begin(), src.end(), back_inserter(dst))`。         |

### 示例

使用 `next_permutation` 生成 $1$ 到 $9$ 的全排列。

```c++
int N = 9, a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
do {
	for (int i = 0; i < N; i++) cout << a[i] << " ";
	cout << endl;
} while (next_permutation(a, a + N));
```

使用 `lower_bound` 与 `upper_bound` 查找有序数组 $a$ 中小于 $x$，等于 $x$，大于 $x$) 元素的分界线。

```c++
int N = 10, a[] = {1, 1, 2, 4, 5, 5, 7, 7, 9, 9}, x = 5; 
int i = lower_bound(a, a + N, x) - a, j = upper_bound(a, a + N, x) - a; 
// a[0] ~ a[i - 1] 为小于x的元素， a[i] ~ a[j - 1] 为等于x的元素， 
// a[j] ~ a[N - 1] 为大于x的元素 
cout << i << " " << j << endl;
```

使用 `partial_sum` 求解 $src$ 中元素的前缀和，并存储于 $dst$ 中。

```c++
vector<int> src = {1, 2, 3, 4, 5}, dst; 
// 求解src中元素的前缀和，dst[i] = src[0] + ... + src[i] 
// back_inserter 函数作用在 dst 容器上，提供一个迭代器 
partial_sum(src.begin(), src.end(), back_inserter(dst)); 
for (unsigned int i = 0; i < dst.size(); i++) cout << dst[i] << " ";
```

使用 `lower_bound` 查找有序数组 $a$ 中最接近 $x$ 的元素。

```c++
int N = 10, a[] = {1, 1, 2, 4, 5, 5, 8, 8, 9, 9}, x = 6; 
// lower_bound将返回a中第一个大于等于x的元素的地址，计算出的i为其下标 
int i = lower_bound(a, a + N, x) - a; 
// 在以下两种情况下，a[i] (a中第一个大于等于x的元素) 即为答案： 
// 1. a中最小的元素都大于等于x； 
// 2. a中存在大于等于x的元素，且第一个大于等于x的元素 (a[i]) 
// 相比于第一个小于x的元素 (a[i - 1]) 更接近x； 
// 否则，a[i - 1] (a中第一个小于x的元素) 即为答案 
if (i == 0 || (i < N && a[i] - x < x - a[i - 1])) 
	cout << a[i]; 
else 
	cout << a[i - 1];
```

使用 `sort` 与 `unique` 查找数组 $a$ 中 **第 $k$ 小的值**（注意：重复出现的值仅算一次，因此本题不是求解第 $k$ 小的元素）。

```c++
int N = 10, a[] = {1, 3, 3, 7, 2, 5, 1, 2, 4, 6}, k=3;
sort(a, a + N);
// unique 将返回去重之后数组最后一个元素之后的地址，计算出的cnt为去重后数组的长度
int cnt = unique(a, a + N) - a;
cout << a[k-1];
```


# 字符串标准库

## C 标准库

| 函数                                                        | 作用                                                                                                                                          |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `printf("%s", s)`                                         | 用 `%s` 来输出一个字符串（字符数组）                                                                                                                       |
| `scanf("%s", &s)`                                         | 用 `%s` 来读入一个字符串（字符数组）                                                                                                                       |
| `sscanf(const char *__source, const char *__format, ...)` | 从字符串 `__source` 里读取变量，比如 `sscanf(str,"%d",&a)`。                                                                                             |
| `sprintf(char *__stream, const char *__format, ...)`      | 将 `__format` 字符串里的内容输出到 `__stream` 中，比如 `sprintf(str,"%d",i)`。                                                                              |
| `strlen(const char *str)`                                 | 返回从 `str[0]` 开始直到 `'\0'` 的字符数。注意，未开启 O2 优化时，该操作写在循环条件中复杂度是 $\Theta(N)$ 的。                                                                   |
| `strcmp(const char *str1, const char *str2)`              | 按照字典序比较 `str1 str2` 若 `str1` 字典序小返回负值，两者一样返回 `0`，`str1` 字典序更大则返回正值。请注意，不要简单的认为返回值只有 `0`、`1`、`-1` 三种，在不同平台下的返回值都遵循正负，但并非都是 `0`、`1`、`-1`。<br> |
| `strcpy(char *str, const char *src)`                      | 把 `src` 中的字符复制到 `str` 中，`str` `src` 均为字符数组头指针，返回值为 `str` 包含空终止符号 `'\0'`。                                                                    |
| `strncpy(char *str, const char *src, int cnt)`            | 复制至多 `cnt` 个字符到 `str` 中，若 `src` 终止而数量未达 `cnt` 则写入空字符到 `str` 直至写入总共 `cnt` 个字符。                                                               |
| `strcat(char *str1, const char *str2)`                    | 将 `str2` 接到 `str1` 的结尾，用 `*str2` 替换 `str1` 末尾的 `'\0'` 返回 `str1`。                                                                            |
| `strstr(char *str1, const char *str2)`                    | 若 `str2` 是 `str1` 的子串，则返回 `str2` 在 `str1` 的首次出现的地址；如果 `str2` 不是 `str1` 的子串，则返回 `NULL`。                                                      |
| `strchr(const char *str, int c)`                          | 找到在字符串 `str` 中第一次出现字符 `c` 的位置，并返回这个位置的地址。如果未找到该字符则返回 `NULL`。                                                                                |
| `strrchr(const char *str, int c)`                         | 找到在字符串 `str` 中最后一次出现字符 `c` 的位置，并返回这个位置的地址。如果未找到该字符则返回 `NULL`。                                                                               |

## C++标准库

C++ 标准库操作字符串对象 [`std::string`](https://oi-wiki.org/lang/csl/string/)，同时也提供对字符数组的兼容。

| 函数                    | 作用                                                                             |
| --------------------- | ------------------------------------------------------------------------------ |
| 重载了赋值运算符 `+`          | 当 `+` 两边是 `string/char/char[]/const char*` 类型时，可以将这两个变量连接，返回连接后的字符串（`string`）。 |
| 赋值运算符 `=`             | 右侧可以是 `const string/string/const char*/char*`                                  |
| 访问运算符 `[cur]`         | 返回 `cur` 位置的引用。                                                                |
| 访问函数 `data()/c_str()` | 返回一个 `const char*` 指针，内容与该 `string` 相同。                                        |
| 容量函数 `size()`         | 返回字符串字符个数                                                                      |
| `find(ch, start = 0)` | 查找并返回从 `start` 开始的字符 `ch` 的位置；                                                 |
| `rfind(ch)`           | 从末尾开始，查找并返回第一个找到的字符 `ch` 的位置（皆从 `0` 开始）（如果查找不到，返回 `-1`）。                       |
| `substr(start, len)`  | 可以从字符串的 `start`（从 `0` 开始）截取一个长度为 `len` 的字符串（缺省 `len` 时代码截取到字符串末尾）。             |
| `append(s)`           | 将 `s` 添加到字符串末尾。                                                                |
| `append(s, pos, n)`   | 将字符串 `s` 中，从 `pos` 开始的 `n` 个字符连接到当前字符串结尾。                                      |
| `replace(pos, n, s)`  | 删除从 `pos` 开始的 `n` 个字符，然后在 `pos` 处插入串 `s`。                                      |
| `erase(pos, n)`       | 删除从 `pos` 开始的 `n` 个字符。                                                         |
| `insert(pos, s)`      | 在 `pos` 位置插入字符串 `s`。                                                           |

`std::string` 重载了比较逻辑运算符，复杂度是 $\Theta(N)$ 的。

# std 常用类型
## `std::string`

`std::string` 是在标准库 `<string>`（注意不是 C 语言中的 `<string.h>` 库）中提供的一个类，本质上是 `std::basic_string<char>` 的别称。

和许多 STL 容器相同，`string` 能动态分配空间，这使得我们可以直接使用 `std::cin` 来输入，但其速度则同样较慢。这一点也同样让我们不必为内存而烦恼。

`string` 的加法运算符可以直接拼接两个字符串或一个字符串和一个字符。和 `std::vector` 类似，`string` 重载了比较运算符，同样是按字典序比较的，所以我们可以直接调用 `std::sort` 对若干字符串进行排序。

在 C 语言里，也有很多字符串的函数，但是它们的参数都是 char 指针类型的，为了方便使用，`string` 有两个成员函数能够将自己转换为 char 指针——`data()`/`c_str()`（它们几乎是一样的，但最好使用 `c_str()`，因为 `c_str()` 保证末尾有空字符，而 `data()` 则不保证），如：

```c++
printf("%s", s); // 编译错误 
printf("%s", s.data()); // 编译通过，但是是 undefined behavior 
printf("%s", s.c_str()); // 一定能够正确输出
```

很多函数都可以返回 string 的长度：

```c++
printf("s 的长度为 %zu", s.size()); 
printf("s 的长度为 %zu", s.length()); 
printf("s 的长度为 %zu", strlen(s.c_str()));
```

> `strlen()` 的复杂度一定是与字符串长度线性相关的。
> 
> `size()` 和 `length()` 的复杂度在 C++98 中没有指定，在 C++11 中被指定为常数复杂度。但在常见的编译器上，即便是 C++98，这两个函数的复杂度也是常数。

> 这三个函数（以及下面将要提到的 `find` 函数）的返回值类型都是 `size_t`（`unsigned long`）。因此，这些返回值不支持直接与负数比较或运算，建议在需要时进行强制转换。

### 寻找某字符(串) 第一次出现的位置

`find(str,pos)` 函数可以用来查找字符串中一个字符/字符串在 `pos`（含）之后第一次出现的位置（若不传参给 `pos` 则默认为 `0`）。如果没有出现，则返回 `string::npos`（被定义为 `-1`，但类型仍为 `size_t`/`unsigned long`）。

```c++
string s = "OI Wiki", t = "OI", u = "i"; 
int pos = 5; 
printf("字符 I 在 s 的 %lu 位置第一次出现\n", s.find('I')); 
printf("字符 a 在 s 的 %lu 位置第一次出现\n", s.find('a')); 
printf("字符 a 在 s 的 %d 位置第一次出现\n", s.find('a'));
printf("字符串 t 在 s 的 %lu 位置第一次出现\n", s.find(t)); 
printf("在 s 中自 pos 位置起字符串 u 第一次出现在 %lu 位置", s.find(u, pos));
```

输出

```txt
字符 I 在 s 的 1 位置第一次出现 
字符 a 在 s 的 18446744073709551615 位置第一次出现 // 即为 size_t(-1)，具体数值与平台有关。 
字符 a 在 s 的 -1 位置第一次出现 // 强制转换为 int 类型则正常输出 -1 
字符串 t 在 s 的 0 位置第一次出现 
在 s 中自 pos 位置起字符串 u 第一次出现在 6 位置
```

### 截取子串

`substr(pos, len)` 函数的参数返回从 `pos` 位置开始截取最多 `len` 个字符组成的字符串（如果从 `pos` 开始的后缀长度不足 `len` 则截取这个后缀）。

```c++
string s = "OI Wiki", t = "OI"; 
printf("从字符串 s 的第四位开始的最多三个字符构成的子串是 %s\n", s.substr(3, 3).c_str()); 
printf("从字符串 t 的第二位开始的最多三个字符构成的子串是 %s", t.substr(1, 3).c_str());
```

输出

```txt
从字符串 s 的第四位开始的最多三个字符构成的子串是 Wik 
从字符串 t 的第二位开始的最多三个字符构成的子串是 I
```

### 插入/删除字符(串)

`insert(index,count,ch)` 和 `insert(index,str)` 是比较常见的插入函数。它们分别表示在 `index` 处连续插入 `count` 次字符串 `ch` 和插入字符串 `str`。

`erase(index,count)` 函数将字符串 `index` 位置开始（含）的 `count` 个字符删除（若不传参给 `count` 则表示删去 `index` 位置及以后的所有字符）。

```c++
string s = "OI Wiki", t = " Wiki"; 
char u = '!'; s.erase(2); 
printf("从字符串 s 的第三位开始删去所有字符后得到的字符串是 %s\n", s.c_str()); 
s.insert(2, t); 
printf("在字符串 s 的第三位处插入字符串 t 后得到的字符串是 %s\n", s.c_str()); 
s.insert(7, 3, u); 
printf("在字符串 s 的第八位处连续插入 3 次字符串 u 后得到的字符串是 %s", s.c_str());
```

输出

```txt
从字符串 s 的第三位开始删去所有字符后得到的字符串是 OI 
在字符串 s 的第三位处插入字符串 t 后得到的字符串是 OI Wiki 
在字符串 s 的第八位处连续插入 3 次字符串 u 后得到的字符串是 OI Wiki!!!
```

### 替换字符(串)

`replace(pos,count,str)` 和 `replace(first,last,str)` 是比较常见的替换函数。它们分别表示将从 `pos` 位置开始 `count` 个字符的子串替换为 `str` 以及将以 `first` 开始（含）、`last` 结束（不含）的子串替换为 `str`，其中 `first` 和 `last` 均为迭代器。

```c++
string s = "OI Wiki"; s.replace(2, 5, ""); 
printf("将字符串 s 的第 3~7 位替换为空串后得到的字符串是 %s\n", s.c_str()); 
s.replace(s.begin(), s.begin() + 2, "NOI"); 
printf("将字符串 s 的前两位替换为 NOI 后得到的字符串是 %s", s.c_str());
```

输出

```txt
将字符串 s 的第 3~7 位替换为空串后得到的字符串是 OI 
将字符串 s 的前两位替换为 NOI 后得到的字符串是 NOI
```

## `std::pair`

通过灵活使用 `pair`，可以轻松应对 **需要将关联数据捆绑存储、处理** 的场景。

> 与自定义的 `struct` 相比，`pair` 不需要额外定义结构与重载运算符，因此使用起来更加简便。
> 
> 然而，自定义 `struct` 的变量命名往往更加清晰（`pair` 只能使用 `first` 与 `second` 访问包含的两个变量）。同时，如果需要将两个以上的变量进行关联，自定义 `struct` 会更加合适。

### 使用方法

可以在定义时直接完成 `pair` 的初始化。

```c++
pair<int, double> p0(1, 2.0);
```

也可以使用先定义，后赋值的方法完成 `pair` 的初始化。

```c++
pair<int, double> p1; 
p1.first = 1; 
p1.second = 2.0;
```

还可以使用 `std::make_pair` 函数。该函数接受两个变量，并返回由这两个变量组成的 `pair`。

```c++
pair<int, double> p2 = make_pair(1, 2.0);
```

通过成员函数 `first` 与 `second`，可以访问 `pair` 中包含的两个变量。

```c++
int i = p0.first; 
double d = p0.second;
```

也可以对其进行修改

```c++
p1.first++;
```

`pair` 已经预先定义了所有的比较运算符，包括 `<`、`>`、`<=`、`>=`、`==`、`!=`。当然，这需要组成 `pair` 的两个变量所属的数据类型定义了 `==` 和/或 `<` 运算符。

其中，`<`、`>`、`<=`、`>=` 四个运算符会先比较两个 `pair` 中的第一个变量，在第一个变量相等的情况下再比较第二个变量。

```c++
if (p2 >= p3) { 
	cout << "do something here" << endl; 
}
```

由于 `pair` 定义了 STL 中常用的 `<` 与 `==`，使得其能够很好的与其他 STL 函数或数据结构配合。比如，`pair` 可以作为 `priority_queue` 的数据类型。

```c++
priority_queue<pair<int,double>> q;
```

可以将 `pair` 的值赋给另一个类型一致的 `pair`。

```c++
p0 = p1;
```

也可以使用 `swap` 函数交换 `pair` 的值。

```c++
swap(p0, p1); 
p2.swap(p3);
```

### 应用举例

#### 离散化

`pair` 可以轻松实现离散化。

我们可以创建一个 `pair` 数组，将原始数据的值作为每个 `pair` 第一个变量，将原始数据的位置作为第二个变量。在排序后，将原始数据值的排名（该值排序后所在的位置）赋给该值原本所在的位置即可。

```c++
// a为原始数据 
pair<int, int> a[MAXN]; 
// ai为离散化后的数据 
int ai[MAXN]; 
for (int i = 0; i < n; i++) { 
	// first为原始数据的值，second为原始数据的位置 
	scanf("%d", &a[i].first); a[i].second = i; 
} 
// 排序 
sort(a, a + n); 
for (int i = 0; i < n; i++) { 
	// 将该值的排名赋给该值原本所在的位置 
	ai[a[i].second] = i; 
}
```

#### Dijkstra

如前所述，`pair` 可以作为 `priority_queue` 的数据类型。

那么，在 Dijkstra 算法的堆优化中，可以使用 `pair` 与 `priority_queue` 维护节点，将节点当前到起点的距离作为第一个变量，将节点编号作为第二个变量。

```c++
priority_queue<pair<int, int>, std::vector<pair<int, int>>, std::greater<pair<int, int>>> q; 

	... while (!q.empty()) { 
	// dis为入堆时节点到起点的距离，i为节点编号 
	int dis = q.top().first, i = q.top().second; 
	q.pop(); 
	... 
}
```

#### pair 与 map

`map` 的是 C++ 中存储键值对的数据结构。很多情况下，`map` 中存储的键值对通过 `pair` 向外暴露。

```c++
map<int, double> m; 
m.insert(make_pair(1, 2.0));
```

## `std::tuple`

### 简介

tuple是一个固定大小的不同类型值的集合，是泛化的std::pair。我们也可以把他当做一个通用的结构体来用，不需要创建结构体又获取结构体的特征，在某些情况下可以取代结构体使程序更简洁，直观。std::tuple理论上可以有无数个任意类型的成员变量，而std::pair只能是2个成员，因此在需要保存3个及以上的数据时就需要使用tuple元组了。

### 创建与初始化

```c++
std::tuple<T1, T2, TN> t1;            
//创建一个空的tuple对象（使用默认构造），它对应的元素分别是T1和T2...Tn类型，采用值初始化。
std::tuple<T1, T2, TN> t2(v1, v2, ... TN);    
//创建一个tuple对象，它的两个元素分别是T1和T2 ...Tn类型; 要获取元素的值需要通过tuple的成员get<Ith>(obj)进行获取(Ith是指获取在tuple中的第几个元素，请看后面具体实例)。
std::tuple<T1&> t3(ref&); // tuple的元素类型可以是一个引用
std::make_tuple(v1, v2); // 像pair一样也可以通过make_tuple进行创建一个tuple对象
```

 tuple的元素类型为引用：

```cpp
std::string name;
std::tuple<string &, int> tpRef(name, 30);
// 对tpRef第一个元素赋值，同时name也被赋值 - 引用
std::get<0>(tpRef) = "Sven";
 
// name输出也是Sven
std::cout << "name: " << name << '\n';
```

### 有关`tuple`元素的操作

#### 等价结构体

```c++
struct person {
    char *m_name;
    char *m_addr;
    int  *m_ages;
};
 
//可以用tuple来表示这样的一个结构类型，作用是一样的。
std::tuple<const char *, const char *, int>
```

#### 获取tuple元素个数

```cpp
// tuple_size
#include <iostream>     // std::cout
#include <tuple>        // std::tuple, std::tuple_size
 
int main ()
{
  std::tuple<int, char, double> mytuple (10, 'a', 3.14);
 
  std::cout << "mytuple has ";
  std::cout << std::tuple_size<decltype(mytuple)>::value;
  std::cout << " elements." << '\n';
 
  return 0;
}
 
//输出结果：
mytuple has 3 elements
```

#### 获取元素的值

```cpp
// tuple_size
#include <iostream>     // std::cout
#include <tuple>        // std::tuple, std::tuple_size
 
int main ()
{
  std::tuple<int, char, double> mytuple (10, 'a', 3.14);
 
  std::cout << "mytuple has ";
  std::cout << std::tuple_size<decltype(mytuple)>::value;
  std::cout << " elements." << '\n';
 
  //获取元素
  std::cout << "the elements is: ";
  std::cout << std::get<0>(mytuple) << " ";
  std::cout << std::get<1>(mytuple) << " ";
  std::cout << std::get<2>(mytuple) << " ";
 
  std::cout << '\n';
 
  return 0;
}
 
//输出结果：
mytuple has 3 elements.
the elements is: 10 a 3.14 
```

   tuple不支持迭代，只能通过元素索引(或tie解包)进行获取元素的值。但是给定的索引必须是在编译器就已经给定，不能在运行期进行动态传递，否则将发生编译错误：

```cpp
for(int i=0; i<3; i++)
    std::cout << std::get<i>(mytuple) << " "; //将引发编译错误
```

#### 获取元素的类型

要想得到元素类型可以通过tuple_element方法获取，如有以下元组对象：

```cpp
std::tuple<std::string, int> tp("Sven", 20);
 
// 得到第二个元素类型
 
std::tuple_element<1, decltype(tp)>::type ages;  // ages就为int类型
 
ages = std::get<1>(tp);
 
std::cout << "ages: " << ages << '\n';
 
//输出结果： 
ages: 20
```

#### 利用tie进行解包元素的值

 如同pair一样也是可以通过tie进行解包tuple的各个元素的值。如下tuple对象有4个元素，通过tie解包将会把这4个元素的值分别赋值给tie提供的4个变量中。

```c++
#include <iostream>
#include <tuple>
#include <utility>
 
int main(int argc, char **argv) {
    std::tuple<std::string, int, std::string, int> tp;
    tp = std::make_tuple("Sven", 25, "Shanghai", 21);
 
    // 定义接收变量
    std::string name;
    std::string addr;
    int ages;
    int areaCode;
 
    std::tie(name, ages, addr, areaCode) = tp;
    std::cout << "Output: " << '\n';
    std::cout << "name: " << name <<", ";
    std::cout << "addr: " << addr << ", ";
    std::cout << "ages: " << ages << ", ";
    std::cout << "areaCode: " << areaCode << '\n';
 
    return 0;
}
 
//输出结果：
Output: 
name: Sven, addr: Shanghai, ages: 25, areaCode: 21
```

但有时候tuple包含的多个元素时只需要其中的一个或两个元素，如此可以通过std::ignore进行变量占位，这样将会忽略提取对应的元素。可以修改上述例程：

```cpp
#include <iostream>
#include <tuple>
#include <utility>
 
int main(int argc, char **argv) {
    std::tuple<std::string, int, std::string, int> tp;
    tp = std::make_tuple("Sven", 25, "Shanghai", 21);
 
    // 定义接收变量
    std::string name;
    std::string addr;
    int ages;
    int areaCode = 110;
 
    std::tie(name, ages, std::ignore, std::ignore) = tp;
    std::cout << "Output: " << '\n';
    std::cout << "name: " << name <<", ";
    std::cout << "addr: " << addr << ", ";
    std::cout << "ages: " << ages << ", ";
    std::cout << "areaCode: " << areaCode << '\n';
 
    return 0;
}
 
//输出结果：
Output: 
name: Sven, addr: , ages: 25, areaCode: 110
```

#### tuple元素的引用

前面已经列举了将引用作为tuple的元素类型。下面通过引用搭配make_tuple()可以提取tuple的元素值，将某些变量值设给它们，并通过改变这些变量来改变tuple元素的值：

```cpp
#include <iostream>
#include <tuple>
#include <functional>
 
int main(int argc, char **agrv) {
 
    std::tuple<std::string, int, float> tp1("Sven Cheng", 77, 66.1);
 
    std::string name;
    int weight;
    float f;
 
    auto tp2 = std::make_tuple(std::ref(name), std::ref(weight), std::ref(f)) = tp1;
 
    std::cout << "Before change: " << '\n';
    std::cout << "name: " << name << ", ";
    std::cout << "weight: " << weight << ", ";
    std::cout << "f: " << f << '\n';
 
    name = "Sven";
    weight = 80;
    f = 3.14;
 
	std::cout << "After change: " << '\n';
	std::cout << "element 1st: " << std::get<0>(tp2) << ", ";
	std::cout << "element 2nd: " << std::get<1>(tp2) << ", ";
	std::cout << "element 3rd: " << std::get<2>(tp2) << '\n';
 
    return 0;
}
 
//输出结果：
Before change: 
name: Sven Cheng, weight: 77, f: 66.1
After change: 
element 1st: Sven, element 2nd: 80, element 3rd: 3.14
```

