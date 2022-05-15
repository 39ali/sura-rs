use rkyv::vec::ArchivedVec;

pub trait BufferData {
    fn as_bytes(&self) -> &[u8];
    fn bsize(&self) -> usize;
}

impl<T> BufferData for Vec<T> {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.as_ptr() as *const u8,
                self.len() * std::mem::size_of::<T>(),
            )
        }
    }

    fn bsize(&self) -> usize {
        std::mem::size_of_val(self.as_slice())
    }
}

impl<T> BufferData for ArchivedVec<T> {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.as_ptr() as *const u8,
                self.len() * std::mem::size_of::<T>(),
            )
        }
    }

    fn bsize(&self) -> usize {
        std::mem::size_of_val(self.as_slice())
    }
}

#[derive(Default)]
pub struct BufferBuilder {
    data: Vec<u8>,
}

impl BufferBuilder {
    pub fn add(&mut self, vals: &impl BufferData) -> u64 {
        let offset = self.data.len();

        // TODO: maybe we shouldn't clone ?
        self.data.extend_from_slice(vals.as_bytes());
        offset as u64
    }

    pub fn data(&self) -> &[u8] {
        self.data.as_bytes()
    }
}
