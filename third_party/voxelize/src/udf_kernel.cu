#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


struct Point3D {
    float x, y, z;
};

struct Triangle {
    Point3D v0, v1, v2;
};
__device__ Point3D cross(const Point3D& v1, const Point3D& v2) {
    Point3D result;
    result.x = v1.y * v2.z - v1.z * v2.y;
    result.y = v1.z * v2.x - v1.x * v2.z;
    result.z = v1.x * v2.y - v1.y * v2.x;
    return result;
}

// Compute the dot product of two vectors
__device__ float dot(const Point3D& v1, const Point3D& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

// Subtract two 3D points (vector subtraction)
__device__ Point3D subtract(const Point3D& p1, const Point3D& p2) {
    Point3D result = {p1.x - p2.x, p1.y - p2.y, p1.z - p2.z};
    return result;
}
__device__ float magnitude(const Point3D &v) {
	    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}
__device__ bool is_identical(const Point3D & p1,  const Point3D & p2){
	Point3D check = subtract(p1, p2);
	if(check.x==0 && check.y == 0 && check.z == 0)
		return true;
	return false;
}

// Compute the squared distance between two points
__device__ float squaredDistance(const Point3D& p1, const Point3D& p2) {
    return (p1.x - p2.x) * (p1.x - p2.x) +
           (p1.y - p2.y) * (p1.y - p2.y) +
           (p1.z - p2.z) * (p1.z - p2.z);
}
__device__ Point3D normalize(Point3D v){
	float len = sqrtf(dot(v, v));
	if (len ==0)
		return v;
	float scale = 1 / len;
	Point3D result = {v.x * scale, v.y * scale, v.z * scale};
       return result;	
}

__device__ float point_to_line_distance(const Point3D &p, const Point3D &v0, const Point3D &v1) {
    // Direction vector of the line
    Point3D d = subtract(v1, v0);

    // Vector from v0 to point p
    Point3D v0_to_p = subtract(p, v0);

    // Scalar projection of v0_to_p onto the direction vector d
    float t = dot(v0_to_p, d) / dot(d, d);

    Point3D closest_point;

    // Check where the projection falls
    if (t < 0) {
        // Projection falls before v0, so the closest point is v0
        closest_point = v0;
    } else if (t > 1) {
        // Projection falls beyond v1, so the closest point is v1
        closest_point = v1;
    } else {
        // Projection falls within the segment, compute the projection point
        closest_point.x = v0.x + t * d.x;
        closest_point.y = v0.y + t * d.y;
        closest_point.z = v0.z + t * d.z;
    }

    // Calculate the distance between p and the closest point
    Point3D closest_to_p = subtract(p, closest_point);
    return magnitude(closest_to_p);
}

// Compute the distance between a point and a triangle face
__device__ float pointToTriangleDistance(const Point3D& queryPoint, const Point3D& v0, const Point3D& v1, const Point3D& v2, bool inverse=false) {
    // Edge vectors
    Point3D edge0 = subtract(v1, v0);
    Point3D edge1 = subtract(v2, v0);
    if (is_identical(v0, v1) && is_identical(v0, v2))
	    return sqrtf(squaredDistance(queryPoint, v0));
    if (is_identical(v0, v1))
	    return point_to_line_distance(queryPoint, v0, v2);
    if (is_identical(v0, v2))
	    return point_to_line_distance(queryPoint, v0, v1);
    if (is_identical(v1, v2))
	    return point_to_line_distance(queryPoint, v0, v1);
    // Normal vector to the triangle plane
    Point3D normal = cross(edge0, edge1);
    if (inverse)
        normal = cross(edge1, edge0);
    
    // Vector from v0 to queryPoint
    Point3D queryVec = subtract(queryPoint, v0);
    if (dot(normal, normal)==0)
	    return sqrtf(dot(queryVec, queryVec));
    normal = normalize(normal);
    //return 1.0;
    
    // Project the query point onto the triangle's plane
    float distanceToPlane = dot(normal, queryVec); // / sqrtf(dot(normal, normal));
    
// return fabsf(distanceToPlane);
    Point3D projectionPoint = {
        queryPoint.x - distanceToPlane * normal.x,
        queryPoint.y - distanceToPlane * normal.y,
        queryPoint.z - distanceToPlane * normal.z
    };
    // Check if the projection point is inside the triangle using barycentric coordinates
    edge0 = subtract(v0, v1);
    edge1 = subtract(v1, v2);
    Point3D edge2 = subtract(v2, v0);
    Point3D projVec0 = subtract(v0, projectionPoint);
    Point3D projVec1 = subtract(v1, projectionPoint);
    Point3D projVec2 = subtract(v2, projectionPoint);
    Point3D c0 = cross(edge0, projVec0);
    Point3D c1 = cross(edge1, projVec1);
    Point3D c2 = cross(edge2, projVec2);
    if (dot(c0, c1) > 0 && dot(c1, c2) > 0 && dot(c0, c2) > 0)
        return fabsf(distanceToPlane);

    // Otherwise, return the minimum distance to the triangle's edges
    float minEdgeDistance = 1e6f;
    minEdgeDistance = fmin(minEdgeDistance, point_to_line_distance(queryPoint, v0, v1));
    minEdgeDistance = fmin(minEdgeDistance, point_to_line_distance(queryPoint, v0, v2));
    minEdgeDistance = fmin(minEdgeDistance, point_to_line_distance(queryPoint, v1, v2));
    
    
    return minEdgeDistance;
}


__device__ void updateUDF(Triangle t, int* udf, const int DIM, const float threshold) {
    // Compute the bounding box of the triangle
    float minX = fminf(fminf(t.v0.x, t.v1.x), t.v2.x);
    float minY = fminf(fminf(t.v0.y, t.v1.y), t.v2.y);
    float minZ = fminf(fminf(t.v0.z, t.v1.z), t.v2.z);
    float maxX = fmaxf(fmaxf(t.v0.x, t.v1.x), t.v2.x);
    float maxY = fmaxf(fmaxf(t.v0.y, t.v1.y), t.v2.y);
    float maxZ = fmaxf(fmaxf(t.v0.z, t.v1.z), t.v2.z);

    // Convert bounding box to grid coordinates
    int iMin = max(0, (int)floorf((minX + 0.5)  * (DIM-1)));
    int jMin = max(0, (int)floorf((minY + 0.5)  * (DIM-1)));
    int kMin = max(0, (int)floorf((minZ + 0.5)  * (DIM-1)));
    int iMax = min(DIM - 1, (int)floorf((maxX + 0.5)  * (DIM-1)));
    int jMax = min(DIM - 1, (int)floorf((maxY + 0.5)  * (DIM-1)));
    int kMax = min(DIM - 1, (int)floorf((maxZ + 0.5)  * (DIM-1)));

    int range = (int)(threshold + 1);
    
    // Make the bounding box larger than the original
    iMax = min(DIM - 1, iMax + range);
    iMin = max(0, iMin - range);
    jMax = min(DIM - 1, jMax + range);
    jMin = max(0, jMin - range);
    kMax = min(DIM - 1, kMax + range);
    kMin = max(0, kMin - range);

    // Update the valid grids within the bounding box
    for (int i = iMin; i <= iMax; ++i) {
        for (int j = jMin; j <= jMax; ++j) {
            for (int k = kMin; k <= kMax; ++k) {
                int idx = i * DIM * DIM + j * DIM + k;
        
        // Compute the distance from the query point to the triangle
                Point3D queryPoint = {(float)i/(DIM-1) - 0.5, (float)j/(DIM-1) - 0.5, (float)k/(DIM-1) -0.5};
                float distance = pointToTriangleDistance(queryPoint, t.v0, t.v1, t.v2);
                float distance2 = pointToTriangleDistance(queryPoint, t.v0, t.v1, t.v2, true);
                if (distance < threshold / DIM || distance2 < threshold / DIM){
                //distance = distance2;
                    int int_dist = (int)(distance * 10000000);
                    atomicMin(&udf[idx], int_dist);
                }
            }
    
        }
    }
}

__global__ void compute_udf_kernel(float* vertices, int* faces, int * udf, int numTriangles, const int DIM, const float threshold) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < numTriangles) {
        int f0 = faces[t * 3 + 0];
        int f1 = faces[t * 3 + 1];
        int f2 = faces[t * 3 + 2];
        Point3D v0 = {vertices[f0 * 3 + 0], vertices[f0 * 3 + 1], vertices[f0 * 3 + 2]};
        Point3D v1 = {vertices[f1 * 3 + 0], vertices[f1 * 3 + 1], vertices[f1 * 3 + 2]};
        Point3D v2 = {vertices[f2 * 3 + 0], vertices[f2 * 3 + 1], vertices[f2 * 3 + 2]};
        Triangle triangle = {v0, v1, v2};
        updateUDF(triangle, udf, DIM, threshold);
    }
}

void compute_valid_udf_cuda(float* vertices, int* faces, int* udf, int numTriangles, const int DIM=512, const float threshold=8) {
    int blockSize = 256;
    int gridSize = (numTriangles + blockSize - 1) / blockSize;

    // Launch the kernel
    compute_udf_kernel<<<gridSize, blockSize>>>(vertices, faces, udf, numTriangles, DIM, threshold);
}
