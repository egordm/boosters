/// Build a `Tree<ScalarLeaf>` with a compact macro syntax.
///
/// This macro provides a concise way to define test trees without string parsing.
///
/// # Syntax
///
/// - Leaf: `node => leaf(value)`
/// - Numeric split (default left): `node => num(feature, threshold, L) -> left, right`
/// - Numeric split (default right): `node => num(feature, threshold, R) -> left, right`
/// - Categorical split: `node => cat(feature, [cats...], L|R) -> left, right`
///   - Listed categories go **right**.
///
/// # Example
///
/// ```
/// use boosters::{scalar_tree, repr::gbdt::ScalarLeaf};
///
/// let tree = scalar_tree! {
///     0 => num(0, 0.5, L) -> 1, 2,
///     1 => leaf(1.0),
///     2 => leaf(2.0),
/// };
///
/// assert_eq!(tree.predict_row(&[0.3]), &ScalarLeaf(1.0));
/// assert_eq!(tree.predict_row(&[0.7]), &ScalarLeaf(2.0));
/// ```
#[macro_export]
macro_rules! scalar_tree {
    // Entry point
    ($($node_id:literal => $kind:ident $args:tt $(-> $left:literal, $right:literal)?),+ $(,)?) => {{
        #[allow(unused_imports)]
        use $crate::repr::gbdt::{MutableTree, ScalarLeaf, categories_to_bitset};

        let max_id: u32 = [$($node_id as u32),+].into_iter().max().unwrap_or(0);
        let n_nodes = (max_id + 1) as usize;

        let mut tree = MutableTree::<ScalarLeaf>::with_capacity(n_nodes);
        tree.init_root_with_n_nodes(n_nodes);

        $(
            $crate::scalar_tree!(@node tree, $node_id, $kind $args $(-> $left, $right)?);
        )+

        tree.freeze()
    }};

    // Leaf node: ID => leaf(VALUE)
    (@node $tree:ident, $node_id:literal, leaf ($value:expr)) => {
        $tree.make_leaf($node_id as u32, ScalarLeaf($value as f32));
    };

    // Numeric split default left: ID => num(FEATURE, THRESHOLD, L) -> LEFT, RIGHT
    (@node $tree:ident, $node_id:literal, num ($feature:expr, $threshold:expr, L) -> $left:literal, $right:literal) => {
        $tree.set_numeric_split(
            $node_id as u32,
            $feature as u32,
            $threshold as f32,
            true,
            $left as u32,
            $right as u32,
        );
    };

    // Numeric split default right: ID => num(FEATURE, THRESHOLD, R) -> LEFT, RIGHT
    (@node $tree:ident, $node_id:literal, num ($feature:expr, $threshold:expr, R) -> $left:literal, $right:literal) => {
        $tree.set_numeric_split(
            $node_id as u32,
            $feature as u32,
            $threshold as f32,
            false,
            $left as u32,
            $right as u32,
        );
    };

    // Categorical split default left: ID => cat(FEATURE, [CATS...], L) -> LEFT, RIGHT
    (@node $tree:ident, $node_id:literal, cat ($feature:expr, [$($cat:expr),* $(,)?], L) -> $left:literal, $right:literal) => {
        let cats: &[u32] = &[$($cat as u32),*];
        let bitset = categories_to_bitset(cats);
        $tree.set_categorical_split(
            $node_id as u32,
            $feature as u32,
            bitset,
            true,
            $left as u32,
            $right as u32,
        );
    };

    // Categorical split default right: ID => cat(FEATURE, [CATS...], R) -> LEFT, RIGHT
    (@node $tree:ident, $node_id:literal, cat ($feature:expr, [$($cat:expr),* $(,)?], R) -> $left:literal, $right:literal) => {
        let cats: &[u32] = &[$($cat as u32),*];
        let bitset = categories_to_bitset(cats);
        $tree.set_categorical_split(
            $node_id as u32,
            $feature as u32,
            bitset,
            false,
            $left as u32,
            $right as u32,
        );
    };
}

/// Create a packed categorical bitset from a list of category indices.
///
/// This is useful for tests that want to build categorical splits.
#[macro_export]
macro_rules! cat_bitset {
    ($($cat:expr),* $(,)?) => {{
        let cats: &[u32] = &[$($cat as u32),*];
        $crate::repr::gbdt::categories_to_bitset(cats)
    }};
}

/// Build a `Tree<ScalarLeaf>` via `MutableTree` (useful when you want Rust control flow).
///
/// For the most concise option in tests, prefer the [`scalar_tree!`] macro.
pub fn scalar_tree_fn(
    build: impl FnOnce(&mut crate::repr::gbdt::MutableTree<crate::repr::gbdt::ScalarLeaf>),
) -> crate::repr::gbdt::Tree<crate::repr::gbdt::ScalarLeaf> {
    let mut t = crate::repr::gbdt::MutableTree::<crate::repr::gbdt::ScalarLeaf>::new();
    build(&mut t);
    t.freeze()
}

/// Same as [`scalar_tree_fn`], but with an explicit capacity.
pub fn scalar_tree_with_capacity(
    capacity: usize,
    build: impl FnOnce(&mut crate::repr::gbdt::MutableTree<crate::repr::gbdt::ScalarLeaf>),
) -> crate::repr::gbdt::Tree<crate::repr::gbdt::ScalarLeaf> {
    let mut t =
        crate::repr::gbdt::MutableTree::<crate::repr::gbdt::ScalarLeaf>::with_capacity(capacity);
    build(&mut t);
    t.freeze()
}
