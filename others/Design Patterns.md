---
date: 2024-11-20 18:39:51
date modified: 2024-11-20 18:45:31
title: Welcome to use Hexo Theme Keep
tags:
  - Hexo
  - Keep
categories:
  - Hexo
---
# What's a design pattern?

**Design patterns** are typical solutions to commonly occurring problems in software design. They are like pre-made blueprints that you can customize to solve a recurring design problem in your code.

What does the pattern consist of?

Most patterns are described very formally so people can reproduce them in many contexts. Here are the sections that are usually present in a pattern description:

- **Intent** of the pattern briefly describes both the problem and the solution.

- **Motivation** further explains the problem and the solution the pattern makes possible.

- **Structure** of classes shows each part of the pattern and how they are related.

- **Code example** in one of the popular programming languages makes it easier to grasp the idea behind the pattern.

Some pattern catalogs list other useful details, such as applicability of the pattern, implementation steps and relations with other patterns.

# Creational Patterns

## Abstract Factory 


```cpp
/**
 * Each distinct product of a product family should have a base interface. All
 * variants of the product must implement this interface.
 */
class AbstractProductA {
public:
	virtual ~AbstractProductA(){};
	virtual std::string UsefulFunctionA() const = 0;
};

/**
 * Concrete Products are created by corresponding Concrete Factories.
 */
 class ConcreteProductA1 : public AbstractProductA {
 public:
	 std::string UsefulFunctionA() const override {
		 return "The result of the product A1.";
	 }
 };


/**
 * Here's the the base interface of another product. All products can interact
 * with each other, but proper interaction is possible only between products of
 * the same concrete variant.
 */
class AbstractProductB {
   /**
   * Product B is able to do its own thing...
   */
 public:
	virtual ~AbstractProductB(){};
	virtual std::string UsefulFunctionB() const = 0;
  /**
   * ...but it also can collaborate with the ProductA.
   *
   * The Abstract Factory makes sure that all products it creates are of the
   * same variant and thus, compatible.
   */
	virtual std::string AnotherUsefulFunctionB(const AbstractProductA &collaborator) const = 0;
 }

/**
 * Concrete Products are created by corresponding Concrete Factories.
 */
class ConcreteProductB1 : public AbstractProductB {
public:
	std::string UsefulFunctionB() const override {
		return "The result of the product B1.";
	}

	/**
   * The variant, Product B1, is only able to work correctly with the variant,
   * Product A1. Nevertheless, it accepts any instance of AbstractProductA as an
   * argument.
   */
   std::string 

}

```


# Structural Patterns



# Behavioral Patterns


