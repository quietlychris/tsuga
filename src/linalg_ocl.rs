use ndarray::prelude::*;

use std::iter::FromIterator;
use std::time::{Duration, Instant};

// Note: From benchmarking, the highest contribution to the runtime of this function is the conversion from an Array2 struct into a vector. In the context of a dense neural network, it's probably possible to do all of that overhead at the beginning, then keep exchanging the already-built vectors back and forth.
use ocl::enums::DeviceSpecifier::*;
use ocl::{Buffer, Device, MemFlags, Platform, ProQue, SpatialDims::*};

struct Context {
    pq: ProQue,
}

pub fn build_ocl_proque(gpu_type: String) -> ProQue {
    let src = include_str!("cl/functions.cl");

    let mut dev = None;
    let platforms = Platform::list();
    for p_idx in 0..platforms.len() {
        let platform = &platforms[p_idx];
        let devices = Device::list_all(platform).unwrap();
        for d_idx in 0..devices.len() {
            let device = devices[d_idx];
            println!("Device: {:?}", device.name());
            if device.name().unwrap().to_string().contains(&gpu_type) {
                dev = Some(device);
            }
            //let deviceinforesult = core::get_device_info(&device, DeviceInfo::MaxComputeUnits);
            //let units = deviceinforesult.to_string().parse().unwrap();
        }
    }

    //println!("The WORK_SIZE is {}",WORK_SIZE);
    let mut ocl_pq = ProQue::builder()
        .src(src)
        .device(dev.unwrap())
        .build()
        .expect("Build ProQue");
    //println!("Built proque: {}",now.elapsed().as_millis());

    println!(
        "The specified device is: {}",
        ocl_pq.device().name().unwrap()
    );
    println!(
        "It has a maximum working group size of {}",
        ocl_pq.device().max_wg_size().unwrap()
    );
    assert!(ocl_pq.device().is_available().unwrap());
    ocl_pq
}

pub fn dot_product(
    ocl_pq: &mut ProQue,
    a: &Vec<f32>,
    b: &Vec<f32>,
    (n, m, k): (usize, usize, usize),
) -> ocl::Result<Vec<f32>> {
    //println!("(n,m,k) = ({},{},{})", n, m, k);

    ocl_pq.set_dims([n, m]);
    //println!("a_vec: {:?}", a_vec);
    let source_buffer_a = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(ocl_pq.dims().clone())
        .copy_host_slice(&a)
        .build()?;
    //println!("Built source_buffer_a");

    ocl_pq.set_dims([m, k]);
    let source_buffer_b = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write())
        .len([m, k])
        .copy_host_slice(&b)
        .build()?;
    // println!("Built source_buffer_b");

    ocl_pq.set_dims([n, k]);
    let result_buffer: Buffer<f32> = ocl_pq.create_buffer()?;
    // println!("The result buffer length is: {}", result_buffer.len());

    // Create a kernel with arguments corresponding to those in the kernel.
    // Just for fun, one argument will be 'named':
    let mut kern = ocl_pq
        .kernel_builder("dot_product")
        .arg(&source_buffer_a)
        .arg(&source_buffer_b)
        .arg(&result_buffer)
        .arg(&m)
        .arg(&k)
        .build()?;

    kern.set_default_global_work_size(Two(n, k)); // This one alone works for MNIST-size sets

    // println!("Kernel global work size: {:?}", kern.default_global_work_size());
    // println!("Kernel local work size: {:?}", kern.default_local_work_size());

    // Enqueue kernel:
    unsafe {
        kern.enq()?;
    }

    // Read results from the device into result_buffer's local vector:
    let mut vec_result = vec![0.; n * k];
    result_buffer.read(&mut vec_result).enq()?;

    Ok(vec_result)
}

pub fn hadamard(
    ocl_pq: &mut ProQue,
    a: &Vec<f32>,
    b: &Vec<f32>,
    (n, m): (usize, usize),
) -> ocl::Result<Vec<f32>> {
    assert_eq!(a.len(), b.len());
    ocl_pq.set_dims(One(n * m));
    let source_buffer_a = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(ocl_pq.dims().clone())
        .copy_host_slice(&a)
        .build()?;

    let source_buffer_b = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(ocl_pq.dims().clone())
        .copy_host_slice(&b)
        .build()?;

    let result_buffer: Buffer<f32> = ocl_pq.create_buffer()?;

    // Create a kernel with arguments corresponding to those in the kernel.
    // Just for fun, one argument will be 'named':
    let mut kern = ocl_pq
        .kernel_builder("hadamard")
        .arg(&source_buffer_a)
        .arg(&source_buffer_b)
        .arg(&result_buffer)
        .build()?;

    kern.set_default_global_work_size(One(n * m));
    kern.set_default_local_work_size(One(n * m));

    // Enqueue kernel:
    unsafe {
        kern.enq()?;
    }

    // Read results from the device into result_buffer's local vector:
    let mut vec_result = vec![0.; n * m];
    result_buffer.read(&mut vec_result).enq()?;

    Ok(vec_result)
}

pub fn multiply_by_scalar(
    ocl_pq: &mut ProQue,
    input: Vec<f32>,
    coeff: f32,
) -> ocl::Result<Vec<f32>> {
    ocl_pq.set_dims(One(input.len()));
    let source_buffer = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(input.len())
        .copy_host_slice(&input)
        .build()?;

    let mut vec_result = vec![0.0f32; input.len()];
    let result_buffer: Buffer<f32> = ocl_pq.create_buffer()?;

    // Create a kernel with arguments corresponding to those in the kernel.
    // Just for fun, one argument will be 'named':
    let mut kern = ocl_pq
        .kernel_builder("multiply_by_scalar")
        .arg(coeff)
        .arg(None::<&Buffer<f32>>)
        .arg_named("result", None::<&Buffer<f32>>)
        .build()?;

    kern.set_default_global_work_size(One(input.len())); // This one alone works for MNIST-size sets

    kern.set_arg(0, &coeff)?;
    kern.set_arg(1, Some(&source_buffer))?;
    kern.set_arg(2, &result_buffer)?;

    // Enqueue kernel:
    unsafe {
        kern.enq()?;
    }

    result_buffer.read(&mut vec_result).enq()?;

    Ok(vec_result)
}

pub fn transpose(
    ocl_pq: &mut ProQue,
    a: &Vec<f32>,
    (n, m): (usize, usize),
) -> ocl::Result<Vec<f32>> {
    ocl_pq.set_dims(Two(n, m));
    let source_buffer_a = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(ocl_pq.dims().clone())
        .copy_host_slice(&a)
        .build()?;

    let result_buffer: Buffer<f32> = ocl_pq.create_buffer()?;

    // Create a kernel with arguments corresponding to those in the kernel.
    // Just for fun, one argument will be 'named':
    let mut kern = ocl_pq
        .kernel_builder("transpose")
        .arg(&source_buffer_a)
        .arg(&result_buffer)
        .arg(&n)
        .arg(&m)
        .build()?;

    kern.set_default_global_work_size(Two(n, m));
    kern.set_default_local_work_size(Two(n, m));

    // Enqueue kernel:
    unsafe {
        kern.enq()?;
    }

    // Read results from the device into result_buffer's local vector:
    let mut vec_result = vec![0.; n * m];
    result_buffer.read(&mut vec_result).enq()?;

    Ok(vec_result)
}

pub fn sigmoid(
    ocl_pq: &mut ProQue,
    input: &Vec<f32>,
    (n, m): (usize, usize),
) -> ocl::Result<Vec<f32>> {
    ocl_pq.set_dims(One(input.len()));
    let source_buffer = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(input.len())
        .copy_host_slice(&input)
        .build()?;

    let mut vec_result = vec![0.0f32; input.len()];
    let result_buffer: Buffer<f32> = ocl_pq.create_buffer()?;

    // Create a kernel with arguments corresponding to those in the kernel.
    // Just for fun, one argument will be 'named':
    let mut kern = ocl_pq
        .kernel_builder("sigmoid")
        .arg(&source_buffer)
        .arg(&result_buffer)
        .build()?;

    kern.set_default_global_work_size(One(n * m));

    // Enqueue kernel:
    unsafe {
        kern.enq()?;
    }

    result_buffer.read(&mut vec_result).enq()?;

    Ok(vec_result)
}

pub fn add(
    ocl_pq: &mut ProQue,
    a: &Vec<f32>,
    b: &Vec<f32>,
    (n, m): (usize, usize),
) -> ocl::Result<Vec<f32>> {
    assert_eq!(a.len(), b.len());
    ocl_pq.set_dims(One(n * m));
    let source_buffer_a = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(ocl_pq.dims().clone())
        .copy_host_slice(&a)
        .build()?;

    let source_buffer_b = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(ocl_pq.dims().clone())
        .copy_host_slice(&b)
        .build()?;

    let result_buffer: Buffer<f32> = ocl_pq.create_buffer()?;

    // Create a kernel with arguments corresponding to those in the kernel.
    // Just for fun, one argument will be 'named':
    let mut kern = ocl_pq
        .kernel_builder("add")
        .arg(&source_buffer_a)
        .arg(&source_buffer_b)
        .arg(&result_buffer)
        .build()?;

    kern.set_default_global_work_size(One(n * m));
    kern.set_default_local_work_size(One(n * m));

    // Enqueue kernel:
    unsafe {
        kern.enq()?;
    }

    // Read results from the device into result_buffer's local vector:
    let mut vec_result = vec![0.; n * m];
    result_buffer.read(&mut vec_result).enq()?;

    Ok(vec_result)
}

pub fn subtract(
    ocl_pq: &mut ProQue,
    a: &Vec<f32>,
    b: &Vec<f32>,
    (n, m): (usize, usize),
) -> ocl::Result<Vec<f32>> {
    assert_eq!(a.len(), b.len());
    ocl_pq.set_dims(One(n * m));
    let source_buffer_a = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(ocl_pq.dims().clone())
        .copy_host_slice(&a)
        .build()?;

    let source_buffer_b = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(ocl_pq.dims().clone())
        .copy_host_slice(&b)
        .build()?;

    let result_buffer: Buffer<f32> = ocl_pq.create_buffer()?;

    // Create a kernel with arguments corresponding to those in the kernel.
    // Just for fun, one argument will be 'named':
    let mut kern = ocl_pq
        .kernel_builder("subtract")
        .arg(&source_buffer_a)
        .arg(&source_buffer_b)
        .arg(&result_buffer)
        .build()?;

    kern.set_default_global_work_size(One(n * m));
    kern.set_default_local_work_size(One(n * m));

    // Enqueue kernel:
    unsafe {
        kern.enq()?;
    }

    // Read results from the device into result_buffer's local vector:
    let mut vec_result = vec![0.; n * m];
    result_buffer.read(&mut vec_result).enq()?;

    Ok(vec_result)
}
