#include <assert.h>
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <getopt.h>
#include <omp.h>

//Some useful macros
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define abs(a) ((a) >= 0 ? (a) : -(a))
#define clamp(a, min, max) ((a) < (min) ? (min) : ((a) > (max) ? (max) : (a)))
#define mag2(x, y) ((x) * (x) + (y) * (y))
#define dist2(x1, y1, x2, y2) (((x1) - (x2)) * ((x1) - (x2)) + ((y1) - (y2)) * ((y1) - (y2)))
#define swap(type, a, b) { type __tmp = (a); (a) = (b); (b) = __tmp; }
#define sq(a) ((a) * (a))

#define SUCCESS 1
#define FAILURE 0


unsigned long gcd_ui(unsigned long x, unsigned long y)
{
	unsigned long t;
	if(y < x)
	{
		t = x;
		x = y;
		y = t;
	}
	while(y > 0)
	{
		t = y;
		y = x % y;
		x = t;
	}
	return x;
}

//binomial coefficients used to compute memory requirements for triangle space
unsigned long binomial(unsigned long n, unsigned long k)
{
	unsigned long d, r = 1;
	if(k == 0)
		return 1;
	if(k == 1)
		return n;
	if(k >= n)
		return (k == n);
	if(k > n / 2)
		k = n - k;
	for(d = 1; d <= k; d++)
	{
		r *= n--;
		r /= d;
	}
	return r;
}

//16 bit pixel element type
typedef uint16_t quantum_t;

#define QUANTUM_MAX (0xffffL)
#define QUANTUM_MIN 0
//clamps values to allowed pixel range 0 - 65535
#define clampq(q) clamp(q, QUANTUM_MIN, QUANTUM_MAX)

//compare function used to qsort quantum
int quantum_compare(const void* a, const void* b);
//selects the k'th largest element in a list of quantum
quantum_t quantum_qselect(quantum_t *v, size_t len, size_t k);

//RGB pixel type
typedef struct
{
	quantum_t red, green, blue;
} pixel_t;

/*
 * Image Type
 * 
 * Images are only created in dynamic memory (not on the stack).
 * Stores the image data as a flexible array member.
 */
typedef struct
{
	size_t w; //width
	size_t h; //height
	pixel_t px[];
} image_t;

//Create new black image of given size
image_t* image_new(size_t w, size_t h);

//create new image by copying from
image_t* image_new_copy(const image_t* from);

//will copy src to dest if possible but may reallocate dest if the sizes do not match
image_t* image_copy(image_t* restrict dest, image_t* restrict src);

/*
 * Creates a new image which is mapped to a temp file.
 * The image can then be used like any other image.
 * This allows algorithms like median to load data beyond
 * the limits of physical memory.
 */
image_t* image_mmap_new(image_t* copy);

//use this free function if memory was allocated with image_mmap_new
void image_mmap_free(image_t* img);

//use this free for all other memory
void image_free(image_t*);

//bilinear interpolation
pixel_t image_interpolate(image_t* img, float x, float y);

//subtract dark from img
void image_subtract_dark(image_t* img, image_t* dark);

//add a scaler value to each pixel
void image_add_scaler(image_t* img, float r, float g, float b);

//multiply each pixel by a scaler
void image_mul_scaler(image_t* img, float r, float g, float b);

//add a scaler to shift the image median to med (used for background calibration)
int image_normalize_median(image_t* img, pixel_t med);

//add a scaler to shift the image mean to mean
void image_normalize_mean(image_t* img, pixel_t mean);

//compute average pixel values in image
pixel_t image_mean(image_t* img);

//compute median (may fail due to memory allocations)
int image_median(image_t* img, pixel_t* med);

//write 16bit RGB ppm to file stream
int image_write_ppm(image_t* img, FILE* f);

//create image from pnm file
image_t* image_new_pnm(FILE* f);

//utility function to display image (debugging only)
int image_display(image_t* img);


/*
 * Data structure to store flat field information.
 * The data in an image can be multiplied by the 
 * corresponding multipliers to even out gain
 * on the different pixels.
 */
typedef struct
{
	size_t w;
	size_t h;
	float_t* red_multiplier;
	float_t* green_multiplier;
	float_t* blue_multiplier;
} flat_t;

//create flat from flat frame
flat_t* flat_new(image_t* flat_image);
void flat_free(flat_t* flat);
//apply a flat to an image
void flat_apply(flat_t* flat, image_t* img);

//stores the information on a star
typedef struct
{
	//lum is the luminosity of a star
	float x, y, lum;
} star_t;

//list of stars stored a flexible array member
typedef struct
{
	size_t size;
	star_t stars[];
} stars_t;

//distance squared between two stars
float star_dist2(star_t s1, star_t s2);

//reversed compare of star luminosity
int compare_star_lum_reversed(const void* a, const void* b);

/*
 * Maps all stars / points in an image.
 * A point is any group of connected pixels
 * which are above detection_percentile.  
 * 
 * Returns a stars_t with the stars sorted from brightest to dimmest.
 * Most of the dimmer stars will be noise which is why only the first
 * couple hundred are considered for matching.
 */
stars_t* stars_map(image_t* img, float detection_percentile);
void stars_free(stars_t* st);

/*
 * The matcher works by finding triangles of the same dimensions.
 * triangle_t is a set of three stars.
 * smax is the star which has the two longest sides.
 * smid is the side which connects the shortest and second shortest sides
 * smin is the star which connects to the two shortest sides.
 * 
 * emax, emid, and emin are the edge lengths
 * 
 * smax <--emax--> smid
 * smax <--emid--> smin
 * smid <--emin--> smin
 */
typedef struct
{
	star_t *smax, *smid, *smin;
	float emax, emid, emin;
} triangle_t;

triangle_t triangle_create(star_t* s1, star_t* s2, star_t* s3);

//square sum of the edge lengths (its distance from the origin in triangle space)
float triangle_mag2(triangle_t t);
int compare_triangle_mag(const void* a, const void* b);

/*
 * Matcher is responsible for finding star matches between two star maps.
 */
typedef struct
{
	//triangles in the reference map
	triangle_t* triangles;
	//stars in the reference map
	star_t* stars;
	size_t nstars;
	size_t ntriangles;
} matcher_t;

//pair of stars produced by the matcher
typedef struct
{
	//domain_star is in the reference
	//range_star is in the image being matched
	star_t domain_star, range_star;
	
	//the level of confidence of this match compared to other matches
	unsigned long votes;
} star_pair_t;

//list of matches from strongest to weakest
typedef struct
{
	size_t size;
	star_pair_t pairs[];
} star_pairs_t;

//create new matcher from a reference set of stars (uses only the first n stars to compute triangle space)
matcher_t* matcher_new(stars_t* stars, size_t n);
void matcher_free(matcher_t* m);

/*
 * Finds the closest match for a triangle in the triangle space.
 * Since the triangle space is sorted it will only search a given
 * range.  This is more for performance than accuracy since the matching
 * algorithm is designed to handle errors.
 */
triangle_t* matcher_closest(matcher_t* matcher, triangle_t* t, float mag_search_bound);

//matches stars baded on differential voting
star_pairs_t* matcher_match_pairs(matcher_t* matcher, stars_t* stars, float search_bound);
void star_pairs_free(star_pairs_t* sp);


/*
 * Affine transform matrix
 * 
 * |a b c|
 * |d e f|
 * |0 0 1|
 */
typedef struct
{
	float a, b, c,
		  d, e, f;
} affine_t;

// Uses only translation and rotation to compute best matrix to transfrom from domain to range.
affine_t affine_align2(float domain_x1, float domain_y1, float domain_x2, float domain_y2,
		float range_x1, float range_y1, float range_x2, float range_y2);

//apply the transform to the given coordinates
void affine_apply(affine_t a, float* x, float* y);

//compute the least squared error transform using the use_first_n pairs
affine_t affine_align_pairs(star_pairs_t* pairs, size_t use_first_n);

/*
 * Accumulators are used to produce images which are the result of 
 * a stacking process.
 * 
 * There are different types of accumulators. These "inherit" from
 * the main accumulator structure.
 */
typedef struct accumulator
{
	//function called to add image
	int (*add)(struct accumulator*, image_t*);
	
	//apply transform and add image
	int (*add_transform)(struct accumulator*, image_t*, affine_t);
	
	//compute final result of accumulation
	image_t* (*result)(struct accumulator*);
	
	//destructor
	void (*free)(struct accumulator*);
} accumulator_t;

int accumulator_add(accumulator_t* accum, image_t* img);
int accumulator_add_transform(accumulator_t* accum, image_t* img, affine_t a);
image_t* accumulator_result(accumulator_t* accum);
void accumulator_free(accumulator_t* accum);

typedef int (*accumulator_add_t)(accumulator_t*, image_t*);
typedef int (*accumulator_add_transform_t)(accumulator_t*, image_t*, affine_t);
typedef image_t* (*accumulator_result_t)(accumulator_t*);
typedef void (*accumulator_free_t)(accumulator_t*);

/*
 * Accumulator to compute mean with constant memory
 */
typedef struct
{
	//make sure it extends the base as the first member of the struct
	accumulator_t base;
	size_t w, h;
	uint32_t* data;
	uint16_t* count;
} mean_accumulator_t;

//virtual function overloads to be assigned to the base function pointers
mean_accumulator_t* mean_accumulator_new();
int mean_accumulator_allocate(mean_accumulator_t* accum, size_t w, size_t h);
int mean_accumulator_add(mean_accumulator_t* accum, image_t* img);
int mean_accumulator_add_transfrom(mean_accumulator_t* accum, image_t* img, affine_t t);
image_t* mean_accumulator_result(mean_accumulator_t* accum);
void mean_accumulator_free(mean_accumulator_t* accum);

/*
 * Computes the median of a series of images.
 * The add function will mmap the images to a temp
 * file so that they may be accessed later.
 */
typedef struct
{
	accumulator_t base;
	size_t w, h;
	bool make_temps; //if false median will simply store all images in memory
	affine_t* transforms;
	image_t** images;
	size_t count, alloc;
	bool calibrate;
} median_accumulator_t;

median_accumulator_t* median_accumulator_new(bool transforms, bool make_temps);
int median_accumulator_add(median_accumulator_t* accum, image_t* img);
int median_accumulator_add_transfrom(median_accumulator_t* accum, image_t* img, affine_t t);
image_t* median_accumulator_result(median_accumulator_t* accum);
void median_accumulator_free(median_accumulator_t* accum);

typedef enum
{
	MEAN, MEDIAN
} stack_algo_t;

stack_algo_t stack_algo_identify(char* algo);
char* stack_algo_tostr(stack_algo_t algo);

typedef struct
{
	size_t alloc;
	size_t size;
	char** file_names;
} file_list_t;

file_list_t* file_list_new();
void file_list_free(file_list_t* lst);
int file_list_add(file_list_t* lst, char* str);

typedef struct
{
	file_list_t* lights;
	file_list_t* darks;
	file_list_t* flats;
	file_list_t* dark_flats;
	file_list_t* bias;
} group_t;

group_t* group_new();
void group_free(group_t* g);
int group_add_flat(group_t* g, char* fn);
int group_add_light(group_t* g, char* fn);
int group_add_dark(group_t* g, char* fn);
int group_add_dark_flat(group_t* g, char* fn);
int group_add_bias(group_t* g, char* fn);

int quantum_compare(const void* a, const void* b)
{
    return (*(quantum_t*)a > *(quantum_t*)b) - (*(quantum_t*)a < *(quantum_t*)b);
}

quantum_t quantum_qselect(quantum_t *v, size_t len, size_t k)
{
	size_t i, st;
	
	for(st = i = 0; i < len - 1; i++)
	{
		if(v[i] > v[len - 1])
			continue;
		swap(quantum_t, v[i], v[st]);
		st++;
	}
	
	swap(quantum_t, v[len - 1], v[st]);
	
	return k == st ? v[st] : st > k ? quantum_qselect(v, st, k) : quantum_qselect(v + st, len - st, k - st);
}

image_t* image_new(size_t w, size_t h)
{
	assert(w && h);
	image_t* img = malloc(sizeof(image_t) + sizeof(pixel_t) * w * h);
	if(!img)
		return NULL;
	
	img->w = w;
	img->h = h;
	
	memset(img->px, 0, sizeof(pixel_t) * w * h);
	
	return img;
}

image_t* image_new_copy(const image_t* from)
{
	assert(from);
	
	image_t* img = image_new(from->w, from->h);
	if(!img)
		return NULL;
	memcpy(img->px, from->px, sizeof(pixel_t) * img->w * img->h);
	return img;
}

image_t* image_copy(image_t* restrict dest, image_t* restrict src)
{
	if(dest->w != src->w || dest->h != src->h)
	{
		image_free(dest);
		dest = image_new(src->w, src->h);
		if(!dest)
			return NULL;
	}
	memcpy(dest->px, src->px, sizeof(pixel_t) * dest->w * dest->h);
	return dest;
}

image_t* image_mmap_new(image_t* img)
{
	FILE* tmp = tmpfile();
	if(tmp == 0)
		return NULL;
	size_t wsize = sizeof(image_t) + sizeof(pixel_t) * img->w * img->h;
	if(fwrite(img, 1, wsize, tmp) != wsize)
	{
		fclose(tmp);
		return NULL;
	}
	
	image_t* nimg = mmap(NULL, wsize, PROT_WRITE | PROT_READ, MAP_SHARED, fileno(tmp), 0);
	fclose(tmp);
	return nimg;
}
void image_mmap_free(image_t* img)
{
	if(img)
	{
		munmap(img, sizeof(image_t) + sizeof(pixel_t) * img->w * img->h);
	}
}

void image_free(image_t* img)
{
	if(img)
	{
		free(img);
	}
}

pixel_t image_interpolate(image_t* img, float x, float y)
{
	size_t w = img->w;
	size_t h = img->h;
	
	size_t x1 = fmaxf(fminf(floorf(x), w - 1), 0);
	size_t x2 = fmaxf(fminf(floorf(x + 1.0f), w - 1), 0);
	size_t y1 = fmaxf(fminf(floorf(y), h - 1), 0);
	size_t y2 = fmaxf(fminf(floorf(y + 1.0f), h - 1), 0);

	
	float c1 = (x2 - x) * (y2 - y);
	float c2 = (x - x1) * (y2 - y);
	float c3 = (x2 - x) * (y - y1);
	float c4 = (x - x1) * (y - y1);
	
	pixel_t q11 = img->px[img->w * y1 + x1];
	pixel_t q12 = img->px[img->w * y2 + x1];
	pixel_t q21 = img->px[img->w * y1 + x2];
	pixel_t q22 = img->px[img->w * y2 + x2];
		
	return (pixel_t)
	{
		c1 * q11.red + c2 * q21.red + c3 * q12.red + c4 * q22.red,
		c1 * q11.green + c2 * q21.green + c3 * q12.green + c4 * q22.green,
		c1 * q11.blue + c2 * q21.blue + c3 * q12.blue + c4 * q22.blue
	};
}

void image_subtract_dark(image_t* img, image_t* dark)
{
	assert(img);
	assert(dark);
	assert(img->w == dark->w && img->h == dark->h);
	
	for(size_t i = 0; i < img->w * img->h; i++)
	{
		img->px[i].red = clampq((int32_t)img->px[i].red - dark->px[i].red);
		img->px[i].green = clampq((int32_t)img->px[i].green - dark->px[i].green);
		img->px[i].blue = clampq((int32_t)img->px[i].blue - dark->px[i].blue);
	}
}

void image_add_scaler(image_t* img, float r, float g, float b)
{
	for(size_t i = 0; i < img->w * img->h; i++)
	{
		img->px[i].red = clampq(img->px[i].red + r);
		img->px[i].green = clampq(img->px[i].green + g);
		img->px[i].blue = clampq(img->px[i].blue + b);
	}
}
void image_mul_scaler(image_t* img, float r, float g, float b)
{
	for(size_t i = 0; i < img->w * img->h; i++)
	{
		img->px[i].red = clampq(img->px[i].red * r);
		img->px[i].green = clampq(img->px[i].green * g);
		img->px[i].blue = clampq(img->px[i].blue * b);
	}
}

int image_normalize_median(image_t* img, pixel_t med)
{
	pixel_t m;
	if(image_median(img, &m) == FAILURE)
		return FAILURE;
	
	image_add_scaler(img, (float)med.red - m.red, 
			(float)med.green - m.green, (float)med.blue - m.blue);
	return SUCCESS;
}

void image_normalize_mean(image_t* img, pixel_t mean)
{
	pixel_t m = image_mean(img);
	image_add_scaler(img, (float)mean.red - m.red, 
			(float)mean.green - m.green, (float)mean.blue - m.blue);
}

pixel_t image_mean(image_t* img)
{
	uint64_t r = 0, g = 0, b = 0;
	for(size_t i = 0; i < img->w * img->h; i++)
	{
		r += img->px[i].red;
		g += img->px[i].green;
		b += img->px[i].blue;
	}
	
	return (pixel_t)
	{
		r / (img->w * img->h),
		g / (img->w * img->h),
		b / (img->w * img->h)
	};
}

int image_median(image_t* img, pixel_t* med)
{
	quantum_t* reds = malloc(sizeof(quantum_t) * img->w * img->h * 3);
	if(reds == NULL)
	{
		return FAILURE;
	}
	quantum_t* greens = reds + img->w * img->h;
	quantum_t* blues = greens + img->w * img->h;
	
	for(size_t i = 0; i < img->w * img->h; i++)
	{
		reds[i] = img->px[i].red;
		greens[i] = img->px[i].green;
		blues[i] = img->px[i].blue;
	}
	
	med->red = quantum_qselect(reds, img->w * img->h, img->w * img->h / 2);
	med->green = quantum_qselect(greens, img->w * img->h, img->w * img->h / 2);
	med->blue = quantum_qselect(blues, img->w * img->h, img->w * img->h / 2);
	
	free(reds);
	
	return SUCCESS;
}

image_t* image_new_pnm(FILE* f)
{
	char id[3];
	if(fscanf(f, "%2s\n", id) != 1)
		return NULL;
	
	if(id[0] != 'P' || id[1] > '6' || id[1] < '1')
		return NULL;
	
	long numbers[3] = {0, 0, 1};
	int n = 0;
	
	while(n < (id[1] == '1' || id[1] == '4' ? 2 : 3))
	{
		int ch;
		while((ch = fgetc(f)) != EOF)
		{
			if(isdigit(ch))
			{
				ungetc(ch, f);
				break;
			}
			else if(ch == '#')
			{
				while((ch = fgetc(f)) != '\n' && ch != EOF);
				if(ch == EOF)
					return NULL;
			}
			else if(!isspace(ch))
				return NULL;
		}
		if(ch == EOF)
			return NULL;
		
		if(fscanf(f, "%ld", numbers + n++) != 1)
			return NULL;
	}
	
	long w = numbers[0];
	long h = numbers[1];
	long mx = numbers[2];
	
	if(w <= 0 || h <= 0 || mx < 1 || mx > 0xffffL)
		return NULL;
	
	if(!isspace(fgetc(f)))
		return NULL;
	
	if(strcmp(id, "P6") == 0)
	{ //binary RGB
		image_t* img = image_new(w, h);
		
		if(mx > 0xff)
		{
			unsigned char buffer[6];
			for(long i = 0; i < w * h; i++)
			{
				if(fread(buffer, 2, 3, f) != 3)
				{
					image_free(img);
					return NULL;
				}
				
				img->px[i] = (pixel_t)
				{
					((quantum_t)buffer[0] << 8) + buffer[1],
					((quantum_t)buffer[2] << 8) + buffer[3],
					((quantum_t)buffer[4] << 8) + buffer[5]
				};
			}
		}
		else
		{
			unsigned char buffer[3];
			for(long i = 0; i < w * h; i++)
			{
				if(fread(buffer, 1, 3, f) != 3)
				{
					image_free(img);
					return NULL;
				}
				
				img->px[i] = (pixel_t)
				{
					((quantum_t)buffer[0] << 8),
					((quantum_t)buffer[1] << 8),
					((quantum_t)buffer[2] << 8)
				};
			}
		}
		return img;
	}
	else if(strcmp(id, "P5") == 0)
	{ //binary GREY
		image_t* img = image_new(w, h);
				
		if(mx > 0xff)
		{
			for(long i = 0; i < w * h; i++)
			{
				quantum_t a = fgetc(f);
				quantum_t b = fgetc(f);
				if(a == EOF || b == EOF)
				{
					image_free(img);
					return NULL;
				}
				quantum_t q = (a << 8) + b;
				img->px[i].red = q;
				img->px[i].green = q;
				img->px[i].blue = q;
			}
		}
		else
		{
			for(long i = 0; i < w * h; i++)
			{
				quantum_t a = fgetc(f);
				if(a == EOF)
				{
					image_free(img);
					return NULL;
				}
				img->px[i].red = a << 8;
				img->px[i].green = a << 8;
				img->px[i].blue = a << 8;
			}
		}
		return img;
	}
	else if(strcmp(id, "P4") == 0)
	{ //binary BITMAP
		image_t* img = image_new(w, h);
		for(long i = 0; i < w * h; i += 8)
		{
			int a = fgetc(f);
			if(a == EOF)
			{
				image_free(img);
				return NULL;
			}
			for(long j = i, k = 7; j < i + 8 && j < w * h; j++, k--)
			{
				quantum_t q = (a >> k) & 1 ? QUANTUM_MAX : QUANTUM_MIN;
				img->px[j].red = q;
				img->px[j].green = q;
				img->px[j].blue = q;
			}
		}
		return img;
	}
	if(strcmp(id, "P3") == 0)
	{ //ascii RGB
		image_t* img = image_new(w, h);
		for(long i = 0; i < w * h; i++)
		{
			long r, g, b;
			if(fscanf(f, "%ld %ld %ld", &r, &g, &b) != 3 || 
					r < 0 || g < 0 || b < 0 || r > mx || g > mx || b > mx)
			{
				image_free(img);
				return NULL;
			}
			if(mx <= 255)
			{
				r <<= 8;
				g <<= 8;
				b <<= 8;
			}
			img->px[i].red = r;
			img->px[i].green = g;
			img->px[i].blue = b;
		}
		return img;
	}
	else if(strcmp(id, "P2") == 0)
	{ //ascii GREY
		image_t* img = image_new(w, h);
		for(long i = 0; i < w * h; i++)
		{
			long g;
			if(fscanf(f, "%ld", &g) != 1 || g < 0 || g > mx)
			{
				image_free(img);
				return NULL;
			}
			if(mx <= 255)
				g <<= 8;
			
			img->px[i].red = g;
			img->px[i].green = g;
			img->px[i].blue = g;
		}
		return img;
	}
	else if(strcmp(id, "P1"))
	{ //ascii BITMAP
		image_t* img = image_new(w, h);
		for(long i = 0; i < w * h; i++)
		{
			long g;
			if(fscanf(f, "%ld", &g) != 1 || g < 0 || g > 1)
			{
				image_free(img);
				return NULL;
			}
			quantum_t q = g ? QUANTUM_MAX : QUANTUM_MIN;
			img->px[i].red = q;
			img->px[i].green = q;
			img->px[i].blue = q;
		}
		return img;
	}
	return NULL;
}

int image_write_ppm(image_t* img, FILE* f)
{
	assert(img);
	
	if(fprintf(f, "P6\n%zu %zu 65535\n", img->w, img->h) <= 0)
		return FAILURE;
	
	unsigned char buffer[6];
	for(size_t i = 0; i < (img->w * img->h); i++)
	{	
		buffer[0] = img->px[i].red >> 8;
		buffer[1] = img->px[i].red & 0xff;
		
		buffer[2] = img->px[i].green >> 8;
		buffer[3] = img->px[i].green & 0xff;
		
		buffer[4] = img->px[i].blue >> 8;
		buffer[5] = img->px[i].blue & 0xff;
		
		if(fwrite(buffer, 1, 6, f) != 6)
			return FAILURE;
	}
	return SUCCESS;
}

int image_display(image_t* img)
{
	FILE* f = popen("display -resize 2000 /dev/stdin", "w");
	if(!f)
		return FAILURE;
	
	if(image_write_ppm(img, f))
		return FAILURE;
	pclose(f);
	
	return SUCCESS;
}

flat_t* flat_new(image_t* flat_image)
{
	assert(flat_image);
	assert(flat_image->w);
	assert(flat_image->h);
	
	flat_t* f = malloc(sizeof(flat_t));
	if(f == NULL)
		return NULL;

	f->w = flat_image->w;
	f->h = flat_image->h;
	
	f->red_multiplier = calloc(flat_image->w * flat_image->h * 3, sizeof(float));
	if(f->red_multiplier == NULL)
	{
		free(f);
		return NULL;
	}
	f->green_multiplier = f->red_multiplier + flat_image->w * flat_image->h;
	f->blue_multiplier = f->green_multiplier + flat_image->w * flat_image->h;
	
	pixel_t mean = image_mean(flat_image);
	
	for(size_t i = 0; i < flat_image->w * flat_image->h; i++)
	{
		f->red_multiplier[i] = (float)mean.red / flat_image->px[i].red;
		f->green_multiplier[i] = (float)mean.green / flat_image->px[i].green;
		f->blue_multiplier[i] = (float)mean.blue / flat_image->px[i].blue;
	}
	
	return f;
}

void flat_free(flat_t* flat)
{
	if(flat)
	{
		free(flat->red_multiplier);
		free(flat);
	}
}

void flat_apply(flat_t* flat, image_t* img)
{
	assert(img->w == flat->w);
	assert(img->h == flat->h);
	
	for(size_t i = 0; i < flat->w * flat->h; i++)
	{
		img->px[i].red = clampq(flat->red_multiplier[i] * img->px[i].red);
		img->px[i].green = clampq(flat->green_multiplier[i] * img->px[i].green);
		img->px[i].blue = clampq(flat->blue_multiplier[i] * img->px[i].blue);
	}
}

float star_dist2(star_t s1, star_t s2)
{
	return dist2(s1.x, s1.y, s2.x, s2.y);
}

int star_compare(const void* a, const void* b)
{
	const star_t* x = a;
	const star_t* y = b;
	
	return (x->lum < y->lum) - (x->lum > y->lum);
}

void stars_free(stars_t* st)
{
	if(st)
	{
		free(st);
	}
}

stars_t* stars_map(image_t* img, float detection_percentile)
{
	assert(img);
	
	quantum_t* grey = NULL;
	quantum_t* sorted = NULL;
	star_t* stars = NULL;
	size_t* visit_stack = NULL;
	size_t visit_alloc = 0;
	size_t nstars = 0;
	size_t stars_alloc = 0;
	
	grey = malloc(sizeof(quantum_t) * img->w * img->h);
	if(grey == NULL) goto fail;
	
	sorted = malloc(sizeof(quantum_t) * img->w * img->h);
	if(sorted == NULL) goto fail;
	
	for(size_t i = 0; i < img->w * img->h; i++)
		sorted[i] = grey[i] = ((uint32_t)img->px[i].red + (uint32_t)img->px[i].green + (uint32_t)img->px[i].blue) / 3;
	
	quantum_t cut = quantum_qselect(sorted, img->w * img->h, img->w * img->h * detection_percentile);
	free(sorted);
	sorted = NULL;
	
	for(size_t i = 0; i < img->w * img->h; i++)
		grey[i] = clampq((int32_t)grey[i] - cut);
		
	visit_stack = malloc(sizeof(size_t) * 1000);
	if(visit_stack == NULL)
		goto fail;
	visit_alloc = 1000;
	
	for(size_t i = 0; i < img->h * img->w; i++)
	{
		if(grey[i])
		{
			if(nstars == stars_alloc)
			{
				stars = realloc(stars, (stars_alloc += 500) * sizeof(star_t));
				if(!stars) goto fail;
			}
			double rowint = 0.0;
			double colint = 0.0;
			double lum = 0.0;
			visit_stack[0] = i;
			
			size_t stack_size = 1;
			while(stack_size)
			{
				size_t gi = visit_stack[--stack_size];
				size_t row = gi / img->w;
				size_t col = gi % img->w;
				
				if(!(row >= 0 && col >= 0 && row < img->h && col < img->w && grey[gi]))
					continue;
				
				rowint += grey[gi] * row;
				colint += grey[gi] * col;
				lum += grey[gi];
				grey[gi] = 0;
				
				for(int dr = -1; dr <= 1; dr++)
				{
					for(int dc = -1; dc <= 1; dc++)
					{
						if(dc == 0 && dr == 0)
							continue;
						if(stack_size == visit_alloc)
						{
							visit_stack = realloc(visit_stack, (visit_alloc *= 2) * sizeof(size_t));
							if(!visit_stack) goto fail;
						}
						visit_stack[stack_size++] = (row + dr) * img->w + (col + dc);
					}
				}
			}
			stars[nstars++] = (star_t){.x = colint / lum, .y = rowint / lum, .lum = lum};
		}
	}
	
	free(visit_stack);
	free(grey);
	
	qsort(stars, nstars, sizeof(star_t), star_compare);
	
	stars_t* result = malloc(sizeof(stars_t) + nstars * sizeof(star_t));
	if(result == NULL) goto fail;
	result->size = nstars;
	memcpy(result->stars, stars, sizeof(star_t) * nstars);
	free(stars);
	
	
	return result;
	
	fail:
	if(grey) free(grey);
	if(sorted) free(sorted);
	if(stars) free(stars);
	if(visit_stack) free(visit_stack);
	
	return NULL;
}

float triangle_mag2(triangle_t t)
{
	return sq(t.emax) + sq(t.emid) + sq(t.emin);
}

triangle_t triangle_create(star_t* s1, star_t* s2, star_t* s3)
{	
	assert(s1 && s2 && s3);
	
	float l1 = star_dist2(*s1, *s2) + star_dist2(*s1, *s3);
	float l2 = star_dist2(*s2, *s1) + star_dist2(*s2, *s3);
	float l3 = star_dist2(*s3, *s1) + star_dist2(*s3, *s2);
	
	if(l1 < l2)
	{
		swap(float, l1, l2);
		swap(star_t*, s1, s2);
	}
	if(l2 < l3)
	{
		swap(float, l2, l3);
		swap(star_t*, s2, s3);
	}
	
	if(l1 < l2)
	{
		swap(float, l1, l2);
		swap(star_t*, s1, s2);
	}
	
	return (triangle_t)
	{
		.emax = sqrt(star_dist2(*s1, *s2)),
		.emid = sqrt(star_dist2(*s1, *s3)),
		.emin = sqrt(star_dist2(*s2, *s3)),
		.smax = s1,
		.smid = s2,
		.smin = s3
	};
}

int compare_triangle_mag(const void* a, const void* b)
{
	const triangle_t* x = a;
	const triangle_t* y = b;
	float mx = triangle_mag2(*x);
	float my = triangle_mag2(*y);
	
	return (mx > my) - (mx < my);
}

void star_pairs_free(star_pairs_t* sp)
{
	if(sp)
	{
		free(sp);
	}
}

matcher_t* matcher_new(stars_t* stars, size_t n)
{
	assert(stars);
	assert(n >= 3 && stars->size >= 3);
	
	n = min(n, stars->size);
	
	matcher_t* matcher = malloc(sizeof(matcher_t));
	if(matcher == NULL)
		return NULL;
	
	matcher->stars = malloc(sizeof(star_t) * n);
	matcher->nstars = n;
	if(matcher->stars == NULL)
	{
		free(matcher);
		return NULL;
	}
	memcpy(matcher->stars, stars->stars, sizeof(star_t) * n);
	
	matcher->ntriangles = binomial(n, 3);
	matcher->triangles = malloc(sizeof(triangle_t) * matcher->ntriangles);
	if(matcher->triangles == NULL)
	{
		free(matcher->stars);
		free(matcher);
		return NULL;
	}
	
	size_t ti = 0;
	for(size_t i = 0; i < matcher->nstars; i++)
	{
		for(size_t j = i + 1; j < matcher->nstars; j++)
		{
			for(size_t k = j + 1; k < matcher->nstars; k++, ti++)
			{
				matcher->triangles[ti] = triangle_create(matcher->stars + i, matcher->stars + j, matcher->stars + k);
			}
		}
	}
	qsort(matcher->triangles, matcher->ntriangles, sizeof(triangle_t), compare_triangle_mag);
	
	return matcher;
}

void matcher_free(matcher_t* m)
{
	if(m)
	{
		free(m->triangles);
		free(m->stars);
		free(m);
	}
}

triangle_t* matcher_closest(matcher_t* matcher, triangle_t* t, float mag_search_bound)
{	
	assert(matcher);
	assert(t);
	
	float lbound = sq(sqrtf(triangle_mag2(*t)) - mag_search_bound);
	float ubound = sq(sqrtf(triangle_mag2(*t)) + mag_search_bound);
	
	triangle_t* start = NULL;
	
	size_t min = 0;
	size_t max = matcher->ntriangles;
	while(min < max)
	{
		size_t mid = (min + max) / 2;
		start = matcher->triangles + mid;
		
		float cmp = triangle_mag2(*start) - lbound;
		
		if(cmp < 0)
			min = mid + 1;
		else if(cmp > 0)
			max = mid;
		else
			break;
	}

	triangle_t* best = NULL;
	float least_err = FLT_MAX;
	triangle_t* it = start;
	
	while(it < (matcher->triangles + matcher->ntriangles) && triangle_mag2(*it) <= ubound)
	{
		float err = sq(it->emax - t->emax) + sq(it->emid - t->emid) + sq(it->emin - t->emin);
		if(err < least_err)
		{
			best = it;
			least_err = err;
		}
		++it;
	}
	
	return best;
}

int star_pair_compare_reversed(const void* a, const void* b)
{
	return (((star_pair_t*)a)->votes < ((star_pair_t*)b)->votes) - (((star_pair_t*)a)->votes > ((star_pair_t*)b)->votes);
}

star_pairs_t* matcher_match_pairs(matcher_t* matcher, stars_t* stars, float search_bound)
{
	assert(matcher);
	assert(stars);
	size_t n = min(matcher->nstars, stars->size);
	
	long* votes = calloc(n * matcher->nstars, sizeof(long));
	if(votes == NULL)
		return NULL;
	
	star_pairs_t* pairs = malloc(sizeof(star_pairs_t) + min(n, matcher->nstars) * sizeof(star_pair_t));
	if(pairs == NULL)
	{
		free(votes);
		return NULL;
	}
	pairs->size = 0;

	for(size_t i = 0; i < n; i++)
	{
		for(size_t j = i + 1; j < n; j++)
		{
			for(size_t k = j + 1; k < n; k++)
			{
				triangle_t t = triangle_create(stars->stars + i, stars->stars + j, stars->stars + k);
				triangle_t* best = matcher_closest(matcher, &t, 3);
				if(best != NULL)
				{
					votes[(t.smax - stars->stars) * matcher->nstars + (best->smax - matcher->stars)]++;
					votes[(t.smid - stars->stars) * matcher->nstars + (best->smid - matcher->stars)]++;
					votes[(t.smin - stars->stars) * matcher->nstars + (best->smin - matcher->stars)]++;
				}
			}
		}
	}
	
	for(size_t i = 0; i < n; i++)
	{
		for(size_t j = 0; j < matcher->nstars; j++)
		{
			long rmax = 0;
			long cmax = 0;
			
			for(size_t k = 0; k < matcher->nstars; k++)
			{
				if(k != j && votes[i * matcher->nstars + k] > rmax)
					rmax = votes[i * matcher->nstars + k];
			}
			
			for(size_t k = 0; k < n; k++)
			{
				if(k != i && votes[k * matcher->nstars + j] > cmax)
					cmax = votes[k * matcher->nstars + j];
			}
			
			long v = votes[i * matcher->nstars + j] - max(rmax, cmax);
			if(v > 0)
			{
				star_pair_t sp;
				sp.domain_star = matcher->stars[j];
				sp.range_star = stars->stars[i];
				sp.votes = v;
				pairs->pairs[pairs->size++] = sp;
			}
		}
	}
	
	free(votes);
	qsort(pairs->pairs, pairs->size, sizeof(star_pair_t), star_pair_compare_reversed);
	return pairs;
}

void affine_apply(affine_t a, float* x, float* y)
{
	float tmp = *x * a.a + *y * a.b + a.c;
	*y = *x * a.d + *y * a.e + a.f;
	*x = tmp;
}

affine_t affine_align2(float domain_x1, float domain_y1, float domain_x2, float domain_y2,
		float range_x1, float range_y1, float range_x2, float range_y2)
{
	float dtheta = atan2f(domain_x1 - domain_x2, domain_y1 - domain_y2) - 
			atan2f(range_x1 - range_x2, range_y1 - range_y2);
	
	affine_t rot = {cosf(dtheta), -sinf(dtheta), 0,
					sinf(dtheta), cosf(dtheta), 0};
	
	float x = domain_x1, y = domain_y1;
	affine_apply(rot, &x, &y);
	
	rot.c = range_x1 - x;
	rot.f = range_y1 - y;
	
	return rot;
}

affine_t affine_align_pairs(star_pairs_t* pairs, size_t use_first_n)
{
	size_t n = min(use_first_n, pairs->size);
	
	float least_err = FLT_MAX;
	affine_t best_a = {1, 0, 0, 0, 1, 0};
	
	for(size_t i = 0; i < n; i++)
	{
		for(size_t j = i + 1; j < n; j++)
		{
			float dx1 = pairs->pairs[i].domain_star.x;
			float dy1 = pairs->pairs[i].domain_star.y;
			float dx2 = pairs->pairs[j].domain_star.x;
			float dy2 = pairs->pairs[j].domain_star.y;
			
			float rx1 = pairs->pairs[i].range_star.x;
			float ry1 = pairs->pairs[i].range_star.y;
			float rx2 = pairs->pairs[j].range_star.x;
			float ry2 = pairs->pairs[j].range_star.y;
			
			affine_t a = affine_align2(dx1, dy1, dx2, dy2, rx1, ry1, rx2, ry2);
			for(size_t k = 0; k < n; k++)
			{
				float x = pairs->pairs[k].domain_star.x, y = pairs->pairs[k].domain_star.y;
				affine_apply(a, &x, &y);
				float d = dist2(x, y, pairs->pairs[k].range_star.x, pairs->pairs[k].range_star.y);
				if(d < least_err)
				{
					least_err = d;
					best_a = a;
				}
			}
		}
	}
	return best_a;
}

int accumulator_add(accumulator_t* accum, image_t* img)
{
	return ((accumulator_t*)accum)->add(accum, img);
}

int accumulator_add_transform(accumulator_t* accum, image_t* img, affine_t a)
{
	return ((accumulator_t*)accum)->add_transform(accum, img, a);
}

image_t* accumulator_result(accumulator_t* accum)
{
	return ((accumulator_t*)accum)->result(accum);
}

void accumulator_free(accumulator_t* accum)
{
	((accumulator_t*)accum)->free(accum);
}

mean_accumulator_t* mean_accumulator_new()
{
	mean_accumulator_t* accum = malloc(sizeof(mean_accumulator_t));
	if(accum == NULL)
		return NULL;
	
	accum->data = NULL;
	accum->w = 0;
	accum->h = 0;
	accum->base.add = (accumulator_add_t)mean_accumulator_add;
	accum->base.add_transform = (accumulator_add_transform_t)mean_accumulator_add_transfrom;
	accum->base.result = (accumulator_result_t)mean_accumulator_result;
	accum->base.free = (accumulator_free_t)mean_accumulator_free;
	return accum;
}

int mean_accumulator_allocate(mean_accumulator_t* accum, size_t w, size_t h)
{
	accum->data = calloc(w * h, 3 * sizeof(uint32_t));
	if(accum->data == NULL)
		return FAILURE;
	accum->count = calloc(w * h, sizeof(uint16_t));
	if(accum->count == NULL)
	{
		free(accum->data);
		return FAILURE;
	}
	
	accum->w = w;
	accum->h = h;
	return SUCCESS;
}


int mean_accumulator_add(mean_accumulator_t* accum, image_t* img)
{
	if(accum->data == NULL)
		mean_accumulator_allocate(accum, img->w, img->h);
	
	for(size_t i = 0; i < accum->h; i++)
	{
		for(size_t j = 0; j < accum->w; j++)
		{
			accum->data[3 * (i * accum->w + j) + 0] += img->px[i * accum->w + j].red;
			accum->data[3 * (i * accum->w + j) + 1] += img->px[i * accum->w + j].green;
			accum->data[3 * (i * accum->w + j) + 2] += img->px[i * accum->w + j].blue;
			accum->count[i * accum->w + j]++;
		}
	}
	return SUCCESS;
}

int mean_accumulator_add_transfrom(mean_accumulator_t* accum, image_t* img, affine_t t)
{
	for(size_t i = 0; i < accum->h; i++)
	{
		for(size_t j = 0; j < accum->w; j++)
		{
			float x = j, y = i;
			
			affine_apply(t, &x, &y);
			if(x < 0 || y < 0 || x > img->w - 1 || y > img->h - 1)
				continue;
			pixel_t p = image_interpolate(img, x, y);
			accum->data[3 * (i * accum->w + j) + 0] += p.red;
			accum->data[3 * (i * accum->w + j) + 1] += p.green;
			accum->data[3 * (i * accum->w + j) + 2] += p.blue;
			accum->count[i * accum->w + j]++;
		}
	}
	return SUCCESS;
}

image_t* mean_accumulator_result(mean_accumulator_t* accum)
{
	image_t* result = image_new(accum->w, accum->h);
	if(result == NULL)
		return NULL;
	for(size_t i = 0; i < accum->h; i++)
	{
		for(size_t j = 0; j < accum->w; j++)
		{
			uint16_t div = accum->count[i * accum->w + j];
			if(div == 0)
				continue;
			result->px[i * accum->w + j].red = accum->data[3 * (i * accum->w + j) + 0] / div;
			result->px[i * accum->w + j].green = accum->data[3 * (i * accum->w + j) + 1] / div;
			result->px[i * accum->w + j].blue = accum->data[3 * (i * accum->w + j) + 2] / div;
		}
	}
	
	return result;
}

void mean_accumulator_free(mean_accumulator_t* accum)
{
	if(accum)
	{
		if(accum->count) free(accum->count);
		if(accum->data) free(accum->data);
		free(accum);
	}
}

median_accumulator_t* median_accumulator_new(bool transforms, bool make_temps)
{
	median_accumulator_t* accum = malloc(sizeof(median_accumulator_t));
	if(accum == NULL)
		return NULL;
	
	accum->images = NULL;
	accum->transforms = NULL;
	accum->count = 0;
	accum->alloc = 0;
	accum->make_temps = make_temps;

	accum->w = 0;
	accum->h = 0;
	accum->base.add = (accumulator_add_t)median_accumulator_add;
	accum->base.add_transform = transforms ? (accumulator_add_transform_t)median_accumulator_add_transfrom : NULL;
	accum->base.result = (accumulator_result_t)median_accumulator_result;
	accum->base.free = (accumulator_free_t)median_accumulator_free;
	return accum;
}

int median_accumulator_add(median_accumulator_t* accum, image_t* img)
{
	if(accum->base.add_transform)
		return median_accumulator_add_transfrom(accum, img, (affine_t){1, 0, 0, 0, 1, 0});
	
	if(accum->count == accum->alloc)
	{
		accum->images = realloc(accum->images, sizeof(image_t*) * (accum->alloc += 10));
		if(accum->images == NULL)
			return FAILURE;
	}
	image_t* mapped = accum->make_temps ? image_mmap_new(img) : image_new_copy(img);
	if(mapped == NULL)
		return FAILURE;
	
	accum->images[accum->count++] = mapped;
	return SUCCESS;
}

int median_accumulator_add_transfrom(median_accumulator_t* accum, image_t* img, affine_t t)
{
	if(accum->count == accum->alloc)
	{
		accum->images = realloc(accum->images, sizeof(image_t*) * (accum->alloc += 10));
		if(accum->images == NULL)
			return FAILURE;
		
		accum->transforms = realloc(accum->transforms, sizeof(affine_t) * accum->alloc);
		if(accum->transforms == NULL)
			return FAILURE;
	}
	image_t* mapped = accum->make_temps ? image_mmap_new(img) : image_new_copy(img);
	if(mapped == NULL)
		return FAILURE;
	accum->transforms[accum->count] = t;
	accum->images[accum->count++] = mapped;
	return SUCCESS;
}

image_t* median_accumulator_result(median_accumulator_t* accum)
{
	image_t* result = image_new(accum->w, accum->h);
	if(result == NULL)
		return NULL;
	bool fail = false;
	#pragma omp parallel
	{
		quantum_t* rlist = malloc(sizeof(quantum_t) * 3 * accum->count);
		if(rlist == NULL)
		{
			#pragma omp atomic write
			fail = true;
		}
		
		#pragma omp barrier
		if(fail)
			goto end;
		
		quantum_t* glist = rlist + accum->count;
		quantum_t* blist = glist + accum->count;
		
		#pragma omp for schedule(dynamic)
		for(size_t i = 0; i < accum->h; i++)
		{
			for(size_t j = 0; j < accum->w; j++)
			{
				size_t len = 0;
				for(size_t k = 0; k < accum->count; k++)
				{
					if(accum->base.add_transform)
					{
						float x = j, y = i;
						affine_apply(accum->transforms[k], &x, &y);
						if(x < 0 || y < 0 || x >= accum->w || y >= accum->h)
							continue;
						pixel_t p = image_interpolate(accum->images[k], x, y);
						rlist[len] = p.red;
						glist[len] = p.green;
						blist[len] = p.blue;
						len++;
					}
					else
					{
						pixel_t p = accum->images[k]->px[i * accum->w + j];
						rlist[len] = p.red;
						glist[len] = p.green;
						blist[len] = p.blue;
						len++;
					}
				}

				if(len == 0)
				{
					result->px[i * accum->w + j] = (pixel_t){0, 0, 0};
				}
				else if(len % 2 == 0)
				{
					uint32_t rplus = quantum_qselect(rlist, len, len / 2);
					uint32_t rminus = quantum_qselect(rlist, len, len / 2 - 1);
					uint32_t gplus = quantum_qselect(glist, len, len / 2);
					uint32_t gminus = quantum_qselect(glist, len, len / 2 - 1);
					uint32_t bplus = quantum_qselect(blist, len, len / 2);
					uint32_t bminus = quantum_qselect(blist, len, len / 2 - 1);
					
					result->px[i * accum->w + j] = (pixel_t)
					{
						(rplus + rminus) / 2,
						(gplus + gminus) / 2,
						(bplus + bminus) / 2
					};
				}
				else
				{
					result->px[i * accum->w + j] = (pixel_t)
					{
						quantum_qselect(rlist, len, len / 2),
						quantum_qselect(glist, len, len / 2),
						quantum_qselect(rlist, len, len / 2)
					};
				}
			}
		}
		end:
		if(rlist) free(rlist);
	}
	
	if(fail)
	{
		image_free(result);
		return NULL;
	}
	return result;
}

void median_accumulator_free(median_accumulator_t* accum)
{
	if(accum)
	{
		if(accum->images)
		{
			for(size_t i = 0; i < accum->count; i++)
			{
				if(accum->make_temps)
					image_mmap_free(accum->images[i]);
				else
					image_free(accum->images[i]);
			}
			free(accum->images);
		}
		if(accum->transforms) free(accum->transforms);
		free(accum);
	}
}



file_list_t* file_list_new()
{
	file_list_t* lst = malloc(sizeof(file_list_t));
	if(lst == NULL)
		return lst;
	lst->alloc = 0;
	lst->size = 0;
	lst->file_names = NULL;
	return lst;
}

void file_list_free(file_list_t* lst)
{
	if(lst)
	{
		if(lst->file_names)
		{
			for(size_t i = 0; i < lst->size; i++)
				free(lst->file_names[i]);
			free(lst->file_names);
		}
		free(lst);
	}
}

int file_list_add(file_list_t* lst, char* str)
{
	if(str == NULL)
		return SUCCESS;
	if(lst->size == lst->alloc)
	{
		lst->file_names = realloc(lst->file_names, sizeof(char*) * (lst->alloc += 10));
		if(lst->file_names == NULL)
			return FAILURE;
	}
	
	lst->file_names[lst->size] = malloc(strlen(str) + 1);
	if(lst->file_names[lst->size] == NULL)
		return FAILURE;
	strcpy(lst->file_names[lst->size++], str);
	return SUCCESS;
}

group_t* group_new()
{
	group_t* g = malloc(sizeof(group_t));
	if(g == NULL)
		return NULL;
	
	g->lights = file_list_new();
	g->darks = file_list_new();
	g->flats = file_list_new();
	g->dark_flats = file_list_new();
	g->bias = file_list_new();
	
	if(g->lights == NULL || g->darks == NULL || g->flats == NULL || g->dark_flats == NULL || g->bias == NULL)
	{
		group_free(g);
		return NULL;
	}
	
	return g;
}
void group_free(group_t* g)
{
	if(g)
	{
		file_list_free(g->lights);
		file_list_free(g->darks);
		file_list_free(g->dark_flats);
		file_list_free(g->flats);
		file_list_free(g->bias);
		free(g);
	}
}
int group_add_flat(group_t* g, char* fn)
{
	return file_list_add(g->flats, fn);
}
int group_add_light(group_t* g, char* fn)
{
	return file_list_add(g->lights, fn);
}
int group_add_dark(group_t* g, char* fn)
{
	return file_list_add(g->darks, fn);
}
int group_add_dark_flat(group_t* g, char* fn)
{
	return file_list_add(g->dark_flats, fn);
}
int group_add_bias(group_t* g, char* fn)
{
	return file_list_add(g->bias, fn);
}

bool quiet = false;
char* script = NULL;
int stack_printf(const char *format, ...);
void stack_assert(bool condition, char* message, ...);
char* fgetline(FILE* f);
image_t* load(char* file);

int stack_printf(const char *format, ...)
{
	if(!quiet)
	{
		va_list args;
		va_start(args, format);
		int r = vfprintf(stderr, format, args);
		va_end(args);
		return r;
	}
	return 0;
}

void stack_assert(bool condition, char* message, ...)
{
	if(!condition)
	{
		if(message == NULL)
			message = "Terminating";
		
		va_list args;
		va_start(args, message);
		vfprintf(stderr, message, args);
		va_end(args);
		exit(1);
	}
}

stack_algo_t stack_algo_identify(char* algo)
{
	if(strcasecmp(algo, "MEAN") == 0)
		return MEAN;
	else if(strcasecmp(algo, "MEDIAN") == 0)
		return MEDIAN;
	
	stack_assert(false, "Invalid Algorithm [%s]\n", algo);
	return MEAN;
}

char* stack_algo_tostr(stack_algo_t algo)
{
	switch(algo)
	{
		case MEAN:
			return "Mean";
		case MEDIAN:
			return "Median";
	}
	return NULL;
}

char* fgetline(FILE* f)
{
	char* ptr = NULL;
	size_t len = 0, alloc = 0;
	
	int ch;
	while((ch = fgetc(f)) != EOF && isspace(ch));
	if(ch == EOF)
		return NULL;
	
	ungetc(ch, f);
	
	while((ch = fgetc(f)) != EOF && ch != '\n')
	{
		if(len == alloc)
		{
			ptr = realloc(ptr, (alloc += 100) + 1);
			stack_assert(ptr, "Out of Memory");
		}
		ptr[len++] = ch;
		ptr[len] = '\0';
	}
	
	if(ptr)
	{
		while(isspace(ptr[len - 1]))
			ptr[--len] = '\0';
	}
	
	return ptr;
}

image_t* load(char* file)
{	
	static long w = -1, h = -1;

	FILE* f = NULL;
	if(script == NULL)
	{
		f =  fopen(file, "r");
		stack_assert(f, "Failed to open file [%s]\n", file);
	}
	else
	{
		char* cmd = alloca(strlen(script) + strlen(file) + 3);
		char* quotes = alloca(strlen(file) + 3);
		sprintf(quotes, "\"%s\"", file);
		sprintf(cmd, script, quotes);
		f = popen(cmd, "r");
		puts(cmd);
		stack_assert(f, "Command [%s] failed.\n", cmd);
		file = cmd;
	}
	
	image_t* img = image_new_pnm(f);
	stack_assert(img, "Failed to read image [%s]\n", file);
	
	if(script)
		pclose(f);
	else
		fclose(f);
	
	if(w == -1 || h == -1)
	{
		w = img->w;
		h = img->h;
	}
	
	stack_assert(img->w == w && img->h == h, "All images must have the same dimensions. [%s]", file);
	return img;
}

void print_usage()
{
	stack_printf("---------- Rapid Stack ----------\n");
	stack_printf("-script [SCRIPT]  Script used for loading image files\n");
	stack_printf("                  Default is to read images in pnm format.\n");
	stack_printf("                  Script must output image in pnm format to stdout.\n");
	stack_printf("                  Insert %%s for the file name of the image in the script.\n");
	stack_printf("                  ie. -script \"dcraw -a -4 -c %%s\"\n\n");
	stack_printf("-lights      [PATH...] Paths to light frames\n");
	stack_printf("-darks       [PATH...] Paths to dark frames\n");
	stack_printf("-flats       [PATH...] Paths to flat frames\n");
	stack_printf("-dark-flats  [PATH...] Paths to dark flat frames\n");
	stack_printf("-group                 Beginning of new calibration group\n");
	stack_printf("-ref                   Marks next light frame as the reference\n\n");
	stack_printf("Algorithm options MEAN, MEDIAN (default is MEAN)\n");
	stack_printf("-lights-algo     [ALGO]    Sets the algorithm to stack all light frames\n");
	stack_printf("-darks-algo      [ALGO]    Sets the algorithm to stack all dark frames\n");
	stack_printf("-flats-algo      [ALGO]    Sets the algorithm to stack all flat frames\n");
	stack_printf("-dark-flats-algo [ALGO]    Sets the algorithm to stack all dark flat frames\n\n");
	stack_printf("-search-bound          [0.0-N]       Sets error tolerance for star match search (default 5.0)\n");
	stack_printf("-nstars              [3-500]       Sets number of stars to be used in alignment (default 100)\n");
	stack_printf("-nmatches            [3-500]       Sets number of matches to be used in alignment (default 15)\n");
	stack_printf("-detection-percentile  [0.95-0.999]  Sets star detection threshold (default 0.99)\n\n");
	stack_printf("-no-temps                   All images are stored in memory and no temporary files are created\n");
	stack_printf("-output  [FILE]              Path of output file (default is stdout)\n");
	stack_printf("-quiet                      No output is written other than errors\n");
	stack_printf("-threads [NUM of THREADS]   Number of threads to use (default is number of CPU cores)\n");
}

int main(int argc, char** argv)
{
	stack_algo_t lights_algo = MEAN;
	stack_algo_t darks_algo = MEAN;
	stack_algo_t flats_algo = MEAN;
	stack_algo_t dark_flats_algo = MEAN;
	stack_algo_t bias_algo = MEAN;
	float search_bound = 5.0f;
	int nstars = 100;
	int nmatches = 10;
	float detection_percentile = 0.99f;
	bool no_temps = false;
	char* output = NULL;
	int threads = omp_get_num_procs();
	char* img_list = NULL;
	
	struct option longopts[] = 
	{
		{"lights-algo", required_argument, NULL, 'l'},
		{"flats-algo", required_argument, NULL, 'f'},
		{"darks-algo", required_argument, NULL, 'd'},
		{"dark-flats-algo", required_argument, NULL, 'z'},
		{"bias-algo", required_argument, NULL, 'b'},
		
		{"search-bound", required_argument, NULL, 'x'},
		{"nmatches", required_argument, NULL, 'n'},
		{"nstars", required_argument, NULL, 'k'},
		{"detection-percentile", required_argument, NULL, 'p'},
		
		{"output", required_argument, NULL, 'o'},
		{"quiet", no_argument, NULL, 'q'},
		{"threads", required_argument, NULL, 'j'},
		
		{"no-temps", no_argument, NULL, 't'},
		{"script", required_argument, NULL, 's'},
		{"image-list", required_argument, NULL, 'i'},
		{"help", no_argument, NULL, 'h'},
		{0, 0, 0, 0}
	};
	
	int c;
	while((c = getopt_long(argc, argv, "l:f:d:z:b:s:n:k:p:o:qj:i:hx:", longopts, NULL)) != -1)
	{
		switch(c)
		{
			case 'l':
				lights_algo = stack_algo_identify(optarg);
				break;
			case 'f':
				flats_algo = stack_algo_identify(optarg);
				break;
			case 'd':
				darks_algo = stack_algo_identify(optarg);
				break;
			case 'z':
				dark_flats_algo = stack_algo_identify(optarg);
				break;
			case 'b':
				bias_algo = stack_algo_identify(optarg);
				break;
			case 'x':
				search_bound = atof(optarg);
				break;
			case 'n':
				nmatches = atoi(optarg);
				break;
			case 'k':
				nstars = atoi(optarg);
				break;
			case 'p':
				detection_percentile = atof(optarg);
				break;
			case 'o':
				output = optarg;
				break;
			case 'q':
				quiet = true;
				break;
			case 'j':
				threads = atoi(optarg);
				break;
			case 'i':
				img_list = optarg;
				break;
			case 's':
				script = optarg;
				break;
			case 'h':
				print_usage();
				break;
			case 't':
				no_temps = true;
				break;
			case ':':
			case '?':
				stack_printf("Try `%s --help' for more information.\n", argv[0]);
				exit(1);
			default:
				stack_printf("%s: invalid option -- %c\n", argv[0], c);
				stack_printf("Try `%s --help' for more information.\n", argv[0]);
				exit(1);
				
		}
	}
	
	stack_assert(img_list, "No image file list given.");
	
	FILE* f = fopen(img_list, "r");
	stack_assert(f, "Could not open image list [%s]\n", img_list);
	
	group_t** groups = NULL;
	size_t group_size = 0;
	bool new_group = true;
	bool next_ref = true;
	bool lights_added = false;
	size_t ref_group = 0;
	size_t ref_index = 0;
	
	enum
	{
		LIGHTS, DARKS, FLATS, BIAS, DFLATS
	} image_type = LIGHTS;
	
	
	char* line = NULL;
	while((line = fgetline(f)) != NULL)
	{
		if(strcmp(line, "-group") == 0)
		{
			new_group = true;
			image_type = LIGHTS;
		}
		else if(strcmp(line, "-lights") == 0)
		{
			image_type = LIGHTS;
		}
		else if(strcmp(line, "-flats") == 0)
		{
			image_type = FLATS;
		}
		else if(strcmp(line, "-darks") == 0)
		{
			image_type = DARKS;
		}
		else if(strcmp(line, "-dark-flats") == 0)
		{
			image_type = DFLATS;
		}
		else if(strcmp(line, "-bias") == 0)
		{
			image_type = BIAS;
		}
		else if(strcmp(line, "-ref") == 0)
		{
			next_ref = true;
		}
		else
		{
			if(new_group)
			{
				groups = realloc(groups, ++group_size * sizeof(group_t*));
				new_group = false;
				stack_assert(groups, "Out of Memory");
				stack_assert((groups[group_size - 1] = group_new()), "Out of Memory");
			}
			
			switch(image_type)
			{
				case LIGHTS:
					stack_assert(group_add_light(groups[group_size - 1], line), "Out of Memory");
					if(next_ref)
					{
						ref_group = group_size - 1;
						ref_index = groups[group_size - 1]->lights->size - 1;
						next_ref = false;
					}
					lights_added = true;
					break;
				case DARKS:
					stack_assert(group_add_dark(groups[group_size - 1], line), "Out of Memory");
					break;
				case FLATS:
					stack_assert(group_add_flat(groups[group_size - 1], line), "Out of Memory");
					break;
				case DFLATS:
					stack_assert(group_add_dark_flat(groups[group_size - 1], line), "Out of Memory");
					break;
				case BIAS:
					stack_assert(group_add_bias(groups[group_size - 1], line), "Out of Memory");
					break;
			}
		}
		free(line);
	}
	fclose(f);
	
	stack_assert(group_size, "No groups to stack.");
	stack_assert(lights_added, "No light images.");
	
	swap(group_t*, groups[0], groups[ref_group]);
	swap(char*, groups[0]->lights->file_names[0], groups[0]->lights->file_names[ref_index]);
	
	stack_printf("Lights Algorithm:     %s\n", stack_algo_tostr(lights_algo));
	stack_printf("Darks Algorithm:      %s\n", stack_algo_tostr(darks_algo));
	stack_printf("Flats Algorithm:      %s\n", stack_algo_tostr(flats_algo));
	stack_printf("Dark Flats Algorithm: %s\n", stack_algo_tostr(dark_flats_algo));
	stack_printf("Bias Algorithm:       %s\n\n", stack_algo_tostr(bias_algo));
	
	stack_printf("Threads: %d\n\n", threads);
	
	stack_printf("%d Groups\n\n", group_size);
	for(size_t i = 0; i < group_size; i++)
	{
		stack_printf("Group %d\n", i + 1);
		stack_printf("\tLights:     %d\n", groups[i]->lights->size);
		stack_printf("\tDarks:      %d\n", groups[i]->darks->size);
		stack_printf("\tFlats:      %d\n", groups[i]->flats->size);
		stack_printf("\tDark Flats: %d\n", groups[i]->dark_flats->size);
		stack_printf("\tBias:       %d\n\n", groups[i]->bias->size);
	}
	
	
	
	for(size_t i = 0; i < group_size; i++)
		group_free(groups[i]);
	free(groups);
	return 0;
}
