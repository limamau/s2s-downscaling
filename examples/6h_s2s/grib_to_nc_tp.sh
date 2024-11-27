for file in *.grib; do
    # Extract the base filename without extension
    base=$(basename "$file" .grib)

    # Copy only tp to a new grib file
    grib_copy -w shortName=tp "$file" "${base}_tp.grib"

    # Run the conversion command
    grib_to_netcdf -o "${base}_tp.nc" "${base}_tp.grib"

    # Remove copied grib file
    rm "${base}_tp.grib"
done