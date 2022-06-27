# SeparatingAxisTheorem_TriangleBox_demo

This demo shows the operation of the Separating Axis Theorem, a theorem used to determine the presence of an intersection between two convex polyhedra, in this case a box and a triangle. 

## Separating Axis Theorem
Two convex polyhedra A and B are disjoint if and only if they can be separated along an axis parallel to the normal vector of a face of A or B or along an axis formed by the vector product of an edge of A, with an edge of B.

### Getting started 
```
python SATdemo.py
```
The program asks the user the coordinates of three vertices of a triangle, the center of th box and the lenght of an edge of the box. It then computes the implementation of the
Separating Axis Theorem, and returns whether the two figures intersect or not
