extern crate ash;
extern crate bitflags;
use bitflags::bitflags;

use core::slice;
use std::{
    borrow::Cow,
    cell::RefCell,
    ffi::{CStr, CString},
};

use ash::{
    extensions::{
        self,
        khr::{Surface, Swapchain},
    },
    vk::{
        self, Image, ImageView, PhysicalDevice, PhysicalDevicePortabilitySubsetFeaturesKHR,
        PhysicalDevicePortabilitySubsetFeaturesKHRBuilder, SwapchainKHR,
    },
    Entry, Instance,
};

use gpu_allocator::vulkan::*;

pub use crate::gpuStructs::*;

#[macro_export]
macro_rules! offset_of {
    ($base:path, $field:ident) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let b: $base = mem::zeroed();
            (&b.$field as *const _ as isize) - (&b as *const _ as isize)
        }
    }};
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) {
        return vk::FALSE;
    }

    println!(
        "{:?}:\n{:?} [{} ({})] : {}\n",
        message_severity,
        message_type,
        message_id_name,
        &message_id_number.to_string(),
        message,
    );

    vk::FALSE
}

pub struct GPUBuffer<'a> {
    pub allocation: Allocation,
    allocator: &'a RefCell<Allocator>,
    device: &'a ash::Device,
    pub buffer: ash::vk::Buffer,
    pub desc: GPUBufferDesc,
}

impl<'a> Drop for GPUBuffer<'a> {
    fn drop(&mut self) {
        unsafe {
            // Cleanup
            self.allocator
                .borrow_mut()
                .free(self.allocation.clone())
                .unwrap();
            self.device.destroy_buffer(self.buffer, None);
        }
    }
}

pub struct GPUImage<'a> {
    allocation: Allocation,
    allocator: &'a RefCell<Allocator>,
    device: &'a ash::Device,
    img: vk::Image,
    format: vk::Format,
    view: vk::ImageView,
}

impl GPUImage<'_> {
    pub fn create_view(
        &self,
        aspect: vk::ImageAspectFlags,
        layer_count: u32,
        level_count: u32,
    ) -> vk::ImageView {
        let depth_image_view_info = vk::ImageViewCreateInfo::builder()
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(aspect)
                    .level_count(layer_count)
                    .layer_count(level_count)
                    .build(),
            )
            .image(self.img)
            .format(self.format)
            .view_type(vk::ImageViewType::TYPE_2D);

        unsafe {
            self.device
                .create_image_view(&depth_image_view_info, None)
                .expect("image view creation failed")
        }
    }
}

impl<'a> Drop for GPUImage<'a> {
    fn drop(&mut self) {
        unsafe {
            // Cleanup
            self.allocator
                .borrow_mut()
                .free(self.allocation.clone())
                .unwrap();
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.img, None);
        }
    }
}

pub struct GFXDevice {
    _entry: Entry,
    instance: ash::Instance,
    pub surface_loader: extensions::khr::Surface,
    pub swapchain_loader: extensions::khr::Swapchain,
    debug_utils_loader: extensions::ext::DebugUtils,
    debug_call_back: vk::DebugUtilsMessengerEXT,

    pub device: ash::Device,
    pub surface: vk::SurfaceKHR,
    pub swapchain: SwapchainData,
    pub pool: vk::CommandPool,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub present_complete_semaphore: vk::Semaphore,
    pub rendering_complete_semaphore: vk::Semaphore,
    pub allocator: RefCell<Allocator>,
    pub graphics_queue: vk::Queue,
}

impl GFXDevice {
    pub fn new(window: &winit::window::Window) -> Self {
        unsafe {
            let entry = Entry::new().unwrap();

            let app_name = CString::new("Sura Engine").unwrap();

            let layer_names = [CString::new("VK_LAYER_KHRONOS_validation").unwrap()];
            let layers_names_raw: Vec<*const i8> = layer_names
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect();

            let surface_extensions = ash_window::enumerate_required_extensions(window).unwrap();

            let mut extension_names_raw = surface_extensions
                .iter()
                .map(|ext| ext.as_ptr())
                .collect::<Vec<_>>();
            extension_names_raw.push(extensions::ext::DebugUtils::name().as_ptr());
            extension_names_raw.push(vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr());

            let appinfo = vk::ApplicationInfo::builder()
                .application_name(&app_name)
                .application_version(0)
                .engine_name(&app_name)
                .engine_version(0)
                .api_version(vk::make_api_version(0, 1, 0, 0));

            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&appinfo)
                .enabled_layer_names(&layers_names_raw)
                .enabled_extension_names(&extension_names_raw);

            let instance = entry
                .create_instance(&create_info, None)
                .expect("Instance creation error");

            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));

            let debug_utils_loader = extensions::ext::DebugUtils::new(&entry, &instance);

            let debug_call_back = debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap();

            let surface_loader = extensions::khr::Surface::new(&entry, &instance);
            let surface = ash_window::create_surface(&entry, &instance, window, None).unwrap();

            let pdevice = GFXDevice::pick_physical_device(&instance, &surface_loader, &surface);
            let device = GFXDevice::create_device(&instance, pdevice.0, pdevice.1 as u32);
            let graphics_queue = device.get_device_queue(pdevice.1 as u32, 0);

            // swapchain
            let size = window.inner_size();
            let swapchain_loader = Swapchain::new(&instance, &device);
            let swapchain = GFXDevice::create_swapchain(
                &pdevice.0,
                &device,
                &surface_loader,
                &surface,
                &swapchain_loader,
                size.width as u32,
                size.height as u32,
            );

            let ci = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(pdevice.1 as u32);

            let pool = device
                .create_command_pool(&ci, None)
                .expect("pool creation failed");

            let ci = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(swapchain.image_count)
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let command_buffers = device.allocate_command_buffers(&ci).unwrap();

            let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            let present_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();
            let rendering_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();

            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device: pdevice.0,
                debug_settings: Default::default(),
                buffer_device_address: true,
            })
            .expect("allocator creation failed");

            Self {
                _entry: entry,
                instance,
                debug_call_back,
                debug_utils_loader,
                surface_loader,
                surface,
                device,
                swapchain_loader,
                swapchain,
                pool,
                command_buffers,
                present_complete_semaphore,
                rendering_complete_semaphore,
                allocator: RefCell::new(allocator),
                graphics_queue,
            }
        }
    }

    pub fn create_image(&self, desc: &GPUImageDesc, data: Option<&[u8]>) -> GPUImage {
        unsafe {
            let img_info = vk::ImageCreateInfo::builder()
                .array_layers(1)
                .mip_levels(1)
                .extent(vk::Extent3D {
                    width: desc.width,
                    height: desc.height,
                    depth: desc.depth,
                })
                .format(vk::Format::R8G8B8A8_SRGB)
                .image_type(vk::ImageType::TYPE_2D)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .samples(vk::SampleCountFlags::TYPE_1)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED);
            // .queue_family_indices(&[]);

            let img = self
                .device
                .create_image(&img_info, None)
                .expect("failed to create image");
            let requirements = self.device.get_image_memory_requirements(img);

            let mut allocation = self
                .allocator
                .borrow_mut()
                .allocate(&AllocationCreateDesc {
                    name: "Image allocation",
                    requirements,
                    location: gpu_allocator::MemoryLocation::CpuToGpu,
                    linear: true, // Buffers are always linear
                })
                .expect("failed to allocate image");

            self.device
                .bind_image_memory(img, allocation.memory(), allocation.offset())
                .unwrap();

            match data {
                Some(content) => {
                    allocation
                        .mapped_slice_mut()
                        .unwrap()
                        .copy_from_slice(content);
                }
                None => {}
            }

            //create view
            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .base_mip_level(0)
                .layer_count(1)
                .level_count(1)
                .build();

            let view_info = vk::ImageViewCreateInfo::builder()
                .format(vk::Format::R8G8B8A8_SNORM)
                .image(img)
                .subresource_range(subresource_range)
                .view_type(vk::ImageViewType::TYPE_2D);

            let view = self
                .device
                .create_image_view(&view_info, None)
                .expect("failed to create image view");

            GPUImage {
                allocation,
                allocator: &self.allocator,
                device: &self.device,
                img,
                format: img_info.format,
                view,
            }
        }
    }

    pub fn create_buffer<T>(&self, desc: &GPUBufferDesc, data: Option<&[T]>) -> GPUBuffer {
        unsafe {
            let mut info = vk::BufferCreateInfo::builder()
                .size(desc.size)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let location = match desc.memory_location {
                MemLoc::CpuToGpu => gpu_allocator::MemoryLocation::CpuToGpu,
                MemLoc::GpuToCpu => gpu_allocator::MemoryLocation::GpuToCpu,
                MemLoc::GpuOnly => gpu_allocator::MemoryLocation::GpuOnly,
                MemLoc::Unknown => gpu_allocator::MemoryLocation::Unknown,
            };

            let mut usage = vk::BufferUsageFlags::default();
            if desc.usage.contains(GPUBufferUsage::TRANSFER_SRC) {
                usage |= vk::BufferUsageFlags::TRANSFER_SRC;
            }
            if desc.usage.contains(GPUBufferUsage::TRANSFER_DST) {
                usage |= vk::BufferUsageFlags::TRANSFER_DST;
            }
            if desc.usage.contains(GPUBufferUsage::VERTEX_BUFFER) {
                usage |= vk::BufferUsageFlags::VERTEX_BUFFER;
            }
            if desc.usage.contains(GPUBufferUsage::INDEX_BUFFER) {
                usage |= vk::BufferUsageFlags::INDEX_BUFFER;
            }
            if desc.usage.contains(GPUBufferUsage::TRANSFER_SRC) {
                usage |= vk::BufferUsageFlags::TRANSFER_SRC;
            }
            if desc.usage.contains(GPUBufferUsage::INDIRECT_BUFFER) {
                usage |= vk::BufferUsageFlags::INDIRECT_BUFFER;
            }

            if desc.usage.contains(GPUBufferUsage::UNIFORM_BUFFER) {
                usage |= vk::BufferUsageFlags::UNIFORM_BUFFER;
            }

            if desc.usage.contains(GPUBufferUsage::STORAGE_BUFFER) {
                usage |= vk::BufferUsageFlags::STORAGE_BUFFER;
            }

            if desc.usage.contains(GPUBufferUsage::STORAGE_TEXEL_BUFFER) {
                usage |= vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER;
            }

            if desc.usage.contains(GPUBufferUsage::UNIFORM_TEXEL_BUFFER) {
                usage |= vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER;
            }
            info.usage = usage;

            let buffer = self.device.create_buffer(&info, None).unwrap();
            let requirements = self.device.get_buffer_memory_requirements(buffer);

            let mut allocation = self
                .allocator
                .borrow_mut()
                .allocate(&AllocationCreateDesc {
                    name: "Buffer allocation",
                    requirements,
                    location,
                    linear: true, // Buffers are always linear
                })
                .unwrap();

            // Bind memory to the buffer
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .unwrap();

            match data {
                Some(content) => {
                    let slice = slice::from_raw_parts(
                        content.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(content),
                    );
                    allocation
                        .mapped_slice_mut()
                        .unwrap()
                        .copy_from_slice(slice);
                }
                None => {}
            }

            GPUBuffer {
                allocation,
                allocator: &self.allocator,
                buffer,
                device: &self.device,
                desc: (*desc).clone(),
            }
        }
    }

    fn create_swapchain(
        pdevice: &PhysicalDevice,
        device: &ash::Device,
        surface_loader: &Surface,
        surface: &vk::SurfaceKHR,
        swapchain_loader: &Swapchain,
        width: u32,
        height: u32,
    ) -> SwapchainData {
        unsafe {
            let surface_format = surface_loader
                .get_physical_device_surface_formats(*pdevice, *surface)
                .unwrap()[0];

            println!("surface format :{:?}", surface_format);

            let surface_capabilities = surface_loader
                .get_physical_device_surface_capabilities(*pdevice, *surface)
                .unwrap();
            let mut desired_image_count = surface_capabilities.min_image_count + 1;
            if surface_capabilities.max_image_count > 0
                && desired_image_count > surface_capabilities.max_image_count
            {
                desired_image_count = surface_capabilities.max_image_count;
            }

            // desired_image_count = 1;
            let surface_resolution = vk::Extent2D::builder().width(width).height(height);
            let pre_transform = if surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_capabilities.current_transform
            };
            let present_modes = surface_loader
                .get_physical_device_surface_present_modes(*pdevice, *surface)
                .unwrap();
            let present_mode = present_modes
                .iter()
                .cloned()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(*surface)
                .min_image_count(desired_image_count)
                .image_color_space(surface_format.color_space)
                .image_format(surface_format.format)
                .image_extent(*surface_resolution)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(pre_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .image_array_layers(1);

            let swapchain = swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .unwrap();

            let present_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
            let present_image_views: Vec<vk::ImageView> = present_images
                .iter()
                .map(|&image| {
                    let create_view_info = vk::ImageViewCreateInfo::builder()
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::R,
                            g: vk::ComponentSwizzle::G,
                            b: vk::ComponentSwizzle::B,
                            a: vk::ComponentSwizzle::A,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image(image);
                    device.create_image_view(&create_view_info, None).unwrap()
                })
                .collect();

            SwapchainData {
                surface_format,
                swapchain,
                image_count: desired_image_count,
                present_images,
                present_image_views,
                width,
                height,
            }
        }
    }

    fn create_device(
        instance: &Instance,
        pdevice: PhysicalDevice,
        queue_family_index: u32,
    ) -> ash::Device {
        unsafe {
            let is_vk_khr_portability_subset = instance
                .enumerate_device_extension_properties(pdevice)
                .unwrap()
                .iter()
                .any(|ext| -> bool {
                    let e = CStr::from_ptr(ext.extension_name.as_ptr());
                    // println!("line : {:?} ", e);
                    if e.eq(vk::KhrPortabilitySubsetFn::name()) {
                        return true;
                    }

                    false
                });

            let mut device_extension_names_raw = vec![Swapchain::name().as_ptr()];

            if is_vk_khr_portability_subset {
                device_extension_names_raw.push(vk::KhrPortabilitySubsetFn::name().as_ptr());
            }

            let features = vk::PhysicalDeviceFeatures {
                shader_clip_distance: 1,
                ..Default::default()
            };

            let priorities = [1.0];

            let queue_info = [vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities)
                .build()];

            let mut ci = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&device_extension_names_raw)
                .enabled_features(&features);

            let mut g: PhysicalDevicePortabilitySubsetFeaturesKHRBuilder;
            if is_vk_khr_portability_subset {
                g = PhysicalDevicePortabilitySubsetFeaturesKHR::builder()
                    .image_view_format_swizzle(true);
                ci = ci.push_next(&mut g);
            }

            //            let mut next = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
            let features2 = &mut vk::PhysicalDeviceFeatures2::builder()
                //              .push_next(&mut next)
                .build();

            instance.get_physical_device_features2(pdevice, features2);

            // println!("next {:?}", next);

            let buffer_address_feature =
                &mut ash::vk::PhysicalDeviceBufferDeviceAddressFeatures::builder()
                    .buffer_device_address(true);
            ci = ci.push_next(buffer_address_feature);

            instance
                .create_device(pdevice, &ci, None)
                .expect("device creation failed")
        }
    }

    fn pick_physical_device(
        instance: &Instance,
        surface_loader: &Surface,
        surface: &vk::SurfaceKHR,
    ) -> (PhysicalDevice, usize) {
        unsafe {
            let pdevices = instance
                .enumerate_physical_devices()
                .expect("physical device error");

            let mut possible_devices = pdevices.iter().filter_map(|pdevice| {
                let props = instance.get_physical_device_queue_family_properties(*pdevice);

                let mut device_match =
                    props
                        .iter()
                        .enumerate()
                        .filter_map(|(queue_family_index, info)| {
                            let mut choose = info.queue_flags.contains(vk::QueueFlags::GRAPHICS);

                            choose = choose
                                && surface_loader
                                    .get_physical_device_surface_support(
                                        *pdevice,
                                        queue_family_index as u32,
                                        *surface,
                                    )
                                    .unwrap();

                            if choose {
                                Some((*pdevice, queue_family_index))
                            } else {
                                None
                            }
                        });

                device_match.next()
            });

            for x in possible_devices.clone() {
                let props = instance.get_physical_device_properties(x.0);

                println!(
                    "device available {:?} , {:?}",
                    CStr::from_ptr(props.device_name.as_ptr()),
                    props.device_type
                );
            }

            let pdevice = possible_devices.next().unwrap();

            let props = instance.get_physical_device_properties(pdevice.0);

            // println!("limits:{:?}", props.limits);

            println!(
                "Picked :{:?} , type:{:?}",
                CStr::from_ptr(props.device_name.as_ptr()),
                props.device_type
            );

            pdevice
        }
    }
}
// create_pipeline();
// create_buffer();
// create_sampler();
// create_texture();

impl Drop for GFXDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            self.swapchain.present_image_views.iter().for_each(|v| {
                self.device.destroy_image_view(*v, None);
            });

            self.device.destroy_command_pool(self.pool, None);

            self.device
                .destroy_semaphore(self.present_complete_semaphore, None);
            self.device
                .destroy_semaphore(self.rendering_complete_semaphore, None);

            self.swapchain_loader
                .destroy_swapchain(self.swapchain.swapchain, None);

            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_call_back, None);

            drop(self.allocator.get_mut());

            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
