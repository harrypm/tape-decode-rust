use std::alloc::{self, Layout};
use std::mem::{self, align_of, size_of, ManuallyDrop};
use std::ptr;

#[inline]
pub(crate) fn convert_vec_in_place<T, U, F>(values: Vec<T>, mut convert: F) -> Vec<U>
where
    F: FnMut(T) -> U,
{
    let size_t = size_of::<T>();
    let size_u = size_of::<U>();

    assert!(
        size_u <= size_t,
        "convert_vec_in_place requires target elements to be no larger than source elements"
    );
    assert!(
        align_of::<U>() <= align_of::<T>(),
        "convert_vec_in_place requires target alignment to be no stricter than source alignment"
    );

    if size_t == 0 || size_u == 0 {
        return values.into_iter().map(convert).collect();
    }

    assert!(
        size_t.is_multiple_of(size_u),
        "convert_vec_in_place requires source element size to be a multiple of target element size"
    );

    let mut values = ManuallyDrop::new(values);
    let ptr_t = values.as_mut_ptr();
    let ptr_u = ptr_t.cast::<U>();
    let len = values.len();
    let capacity = values.capacity();
    let capacity_u = capacity
        .checked_mul(size_t)
        .expect("source Vec allocation size overflowed")
        / size_u;

    struct Guard<T, U> {
        ptr_t: *mut T,
        ptr_u: *mut U,
        len: usize,
        capacity: usize,
        initialized_u: usize,
        next_t: usize,
    }

    impl<T, U> Drop for Guard<T, U> {
        fn drop(&mut self) {
            unsafe {
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                    self.ptr_u,
                    self.initialized_u,
                ));

                for index in self.next_t..self.len {
                    ptr::drop_in_place(self.ptr_t.add(index));
                }

                if self.capacity != 0 {
                    let layout = Layout::array::<T>(self.capacity)
                        .expect("source Vec allocation layout overflowed");
                    alloc::dealloc(self.ptr_t.cast::<u8>(), layout);
                }
            }
        }
    }

    let mut guard = Guard {
        ptr_t,
        ptr_u,
        len,
        capacity,
        initialized_u: 0,
        next_t: 0,
    };

    for index in 0..len {
        unsafe {
            let value = ptr::read(ptr_t.add(index));
            guard.next_t = index + 1;
            ptr::write(ptr_u.add(index), convert(value));
            guard.initialized_u = index + 1;
        }
    }

    mem::forget(guard);

    unsafe { Vec::from_raw_parts(ptr_u, len, capacity_u) }
}
