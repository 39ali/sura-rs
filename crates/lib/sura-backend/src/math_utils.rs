pub fn bitfield_extract(val: u64, offset: u32, bits: u32) -> u64 {
    let max_val = u64::max_value();
    let mask = max_val.rotate_left(bits);

    if offset > 0 {
        (val >> (offset - 1)) & (mask as u64)
    } else {
        val & mask
    }
}
